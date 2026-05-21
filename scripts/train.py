import os
import math
import time
import pickle
import argparse
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from pockliggpt.training.config import load_config
from pockliggpt.training.factories import build_model_classes
from pockliggpt.training.factories import build_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # -------------------
    # config shortcuts
    # -------------------
    out_dir = cfg["logging"]["out_dir"]
    init_ckpt_path = cfg["training"]["init_ckpt_path"]
    init_from = cfg["training"]["init_from"]

    eval_interval = cfg["eval"]["eval_interval"]
    eval_iters = cfg["eval"]["eval_iters"]
    eval_only = cfg["eval"]["eval_only"]
    always_save_checkpoint = cfg["eval"]["always_save_checkpoint"]

    log_interval = cfg["logging"]["log_interval"]

    batch_size = cfg["data"]["batch_size"]
    block_size = cfg["data"]["block_size"]

    n_layer = cfg["model"]["n_layer"]
    n_head = cfg["model"]["n_head"]
    n_embd = cfg["model"]["n_embd"]
    dropout = cfg["model"]["dropout"]
    bias = cfg["model"]["bias"]

    learning_rate = cfg["training"]["learning_rate"]
    max_iters = cfg["training"]["max_iters"]
    weight_decay = cfg["training"]["weight_decay"]
    beta1 = cfg["training"]["beta1"]
    beta2 = cfg["training"]["beta2"]
    grad_clip = cfg["training"]["grad_clip"]
    decay_lr = cfg["training"]["decay_lr"]
    warmup_iters = cfg["training"]["warmup_iters"]
    lr_decay_iters = cfg["training"]["lr_decay_iters"]
    min_lr = cfg["training"]["min_lr"]
    gradient_accumulation_steps = cfg["training"]["gradient_accumulation_steps"]

    backend = cfg["system"]["backend"]
    device = cfg["system"]["device"]
    dtype = cfg["system"]["dtype"]
    compile = cfg["system"]["compile"]

    # -------------------
    # DDP
    # -------------------
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank

        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # -------------------
    # loader
    # -------------------
    loader = build_loader(cfg, device=device, device_type=device_type)

    # -------------------
    # vocab/meta
    # -------------------
    meta_vocab_size = loader.get_vocab_size()

    # -------------------
    # model
    # -------------------
    GPTConfig, GPT = build_model_classes(cfg["model"]["type"])

    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=loader.block_size,
        bias=bias,
        vocab_size=None,
        dropout=dropout,
    )

    iter_num = 0
    best_val_loss = 1e9

    if init_from == "scratch":
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    elif init_from == "resume":
        checkpoint = torch.load(init_ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        ckpt_block_size = checkpoint_model_args["block_size"]
        cfg_block_size = cfg["data"]["block_size"]

        if ckpt_block_size != cfg_block_size:
            raise ValueError(
                f"block_size mismatch: checkpoint has {ckpt_block_size}, "
                f"but config/dataset expect {cfg_block_size}"
            )
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(unwanted_prefix):
                k = k[len(unwanted_prefix):]
            cleaned_state_dict[k] = v

        model.load_state_dict(cleaned_state_dict, strict=True)

    else:
        raise ValueError(f"Unsupported init_from: {init_from}")

    model = model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    if compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                batch = loader.get_batch(split)
                with ctx:
                    _, loss = model(**batch)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def get_lr(it):
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    batch = loader.get_batch("train")
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": cfg,
                    }
                    ckpt_name = cfg["logging"]["checkpoint_name"]
                    torch.save(checkpoint, os.path.join(out_dir, ckpt_name))

        if iter_num == 0 and eval_only:
            break

        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = micro_step == (gradient_accumulation_steps - 1)

            with ctx:
                _, loss = model(**batch)
                loss = loss / gradient_accumulation_steps

            batch = loader.get_batch("train")
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if master_process and iter_num % log_interval == 0:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()