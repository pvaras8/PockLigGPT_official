import os
import pandas as pd
import subprocess
import pickle
import numpy as np
import torch
from model_sequence_pocket_add import GPTConfig, GPT
from tqdm.auto import tqdm
from contextlib import nullcontext
from tqdm.auto import tqdm
# import matplotlib.pylab as plt
from omegaconf import DictConfig
from dataclasses import dataclass
from torchtyping import TensorType
from typing import Iterable, Sequence, List, Tuple
import random
import selfies as sf
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem
from rdkit.Chem import Descriptors


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from sigmoid_function import combine_docking_only

from sigmoid_function import combine_docking_logP_mw
from sigmoid_function import combine_docking_logP
from sigmoid_function import combine_two_docking_specificity
from sigmoid_function import combine_two_docking_specific_2
from sigmoid_function import combine_two_docking_specific_new






# -----------------------------------------------------------------------------
# Configuración inicial

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Definir el dtype globalmente
dtype = torch.bfloat16

train_device = "cuda:0"  # Primera GPU
ref_device   = "cuda:1"  # Segunda GPU
# train_device  = 'cuda' if torch.cuda.is_available() else 'cpu'
# ref_device  = 'cuda' if torch.cuda.is_available() else 'cpu'




# Paths
reward_script = os.path.abspath("multiaffinity_rga_2/reward_model_3.py")  # Absolute path to reward_model_2.py
reward_script_2 = os.path.abspath("multiaffinity_rga_2/reward_model_4.py")  # Absolute path to reward_model_2.py
reward_script_3 = os.path.abspath("multiaffinity_rga_2/reward_model_5.py")  # Absolute path to reward_model_2.py


vars_file = os.path.abspath("multiaffinity_rga_2/vars_2.json")  # Absolute path to vars.json
vars_file_2 = os.path.abspath("multiaffinity_rga_2/vars_3.json")  # Absolute path to vars.json
vars_file_3 = os.path.abspath("multiaffinity_rga_2/vars_4.json")  # Absolute path to vars.json



generated_smi_file = os.path.abspath("generated_molecules_M.csv")  # Absolute path for the output SMILES file
affinity_script = os.path.abspath("affinity/run.py")  # Script for affinity prediction
affinity_model_path = os.path.abspath("affinity/models/dyrk1a/xg_model_dyrk1a.pkl")  # Path to the affinity model
data_path = os.path.abspath("zink_250k.csv")  # Script for rollout

input_smi_file = os.path.abspath("generated_molecule/chemgpt_buche.csv")



exec(open('configurator.py').read())  # Load additional configurations

# -----------------------------------------------------------------------------
args = {
    'batch_size': 64,
    'lr': 3e-5,
    'prompt_size': 51,
    'prompt_batch_size': 256,
    'num_rollouts': 256,
    'epochs': 30,
    'ppo_epochs': 4,
    'gen_kwargs': {
        'seq_length': 156,
        'max_new_tokens': 106,  # Tokens generados después del prompt.
        'top_k': 10,           # Aumenta la diversidad de generación.
        'temperature': 0.9,    # Más exploración en la generación.
    },
    'kl_coef': 0.005,
    'gamma': 0.99,      # Factor de descuento.
    'lam': 0.95,        # GAE lambda.
    'cliprange': 0.2,   # Reducción para evitar oscilaciones grandes.
    'cliprange_value': 0.1,
    'vf_coef': 1,       # Mantiene la importancia del valor estimado.
}



args = DictConfig(args)

# Pocket dado (3 letras, separado por espacios)

# -----------------------------------------------------------------------------
# Cargar modelo
init_from = 'resume'
out_dir = 'out-models'
ckpt_path = os.path.join(out_dir, 'ckpt_zink_chembl_cross_sequence_add.pt')
checkpoint = torch.load(ckpt_path, map_location='cpu')


# -----------------------------------------------------------------------------
# Funciones encode/decode
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta_chembl_db_aa_2_proto4.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[s]] if s in stoi else [stoi[c] for c in s]
        decode = lambda l: ''.join([itos.get(i, '') for i in l if i in itos])


# pocket_str = "SER LEU ILE GLY LYS PHE GLN VAL VAL LYS VAL ALA ILE LYS VAL PHE GLU MET LEU SER TYR ASN LEU ASP PRO GLU ASN ILE LEU LEU LYS ILE VAL ASP"
pocket_str = "ASN ASP SER PHE GLY SER TRP ASN GLY GLY GLY PHE GLN THR GLY TYR GLU SER ALA TRP THR PRO LEU SER VAL GLU GLY THR ALA PHE LEU VAL TYR PHE TRP MET HIS GLY TYR"
# pocket_str = "GLY GLN GLY TYR LEU VAL ASP THR GLY SER TYR THR GLN GLY TRP LYS PHE PHE ILE TRP GLY ILE LEU GLY ASP SER GLY THR THR ASN ARG VAL GLY ALA"

# === Embeddings del pocket (ProtT5 por residuo) ===
POCKET_EMB_PATH = "/LUSTRE/users/pvaras/sequence_gpt/data/very_big_molecules_selfies_2/cross_docked_jblasser/p_buche_per_residue.npy"

pocket_emb_np = np.load(POCKET_EMB_PATH)  # (L_raw, d)

# Obtener nº AA reales directamente de la secuencia
aa_list = [aa.strip().upper() for aa in pocket_str.split()]
L_seq = len(aa_list)

# Recortar: quedarte solo con 1 vector por AA
if pocket_emb_np.shape[0] == L_seq + 1:
    pocket_emb_np = pocket_emb_np[:L_seq]   # quitar </s>
elif pocket_emb_np.shape[0] != L_seq:
    raise ValueError(f"Inconsistencia: {pocket_emb_np.shape[0]} embeddings vs {L_seq} AA")

NUM_RES, D_PROT = pocket_emb_np.shape
print("Pocket BACE emb corregido:", pocket_emb_np.shape)



def pocket_tokens_from_string(pocket_str, stoi):
    aa_list = [aa.strip().upper() for aa in pocket_str.split()]
    return [stoi["<POCKET>"]] + [stoi[f"<AA_{aa}>"] for aa in aa_list] + [stoi["</POCKET>"]]

POCKET_TOKENS = pocket_tokens_from_string(pocket_str, stoi)

def strip_to_ligand(text):
    # 1️⃣ Elimina el pocket completo si está presente
    if "<POCKET>" in text and "</POCKET>" in text:
        text = text.split("</POCKET>", 1)[1]  # corta después del cierre del pocket

    # 2️⃣ Quédate solo con lo que hay dentro del ligando
    if "<LIGAND>" in text:
        text = text.split("<LIGAND>", 1)[1]
    if "</LIGAND>" in text:
        text = text.split("</LIGAND>", 1)[0]

    # 3️⃣ Corta en <EOS> si aparece
    text = text.split("<EOS>")[0]

    # 4️⃣ Limpieza de cualquier marcador residual
    for tok in ["<SOS>", "<POCKET>", "</POCKET>", "<LIGAND>", "</LIGAND>", "<EOS>"]:
        text = text.replace(tok, "")

    return text.strip()

# -----------------------------------------------------------------------------
# Clases auxiliares

class PromptDataset:
    def __init__(self, prompt_size, zinc_data):
        """
        Inicializa el dataset con múltiples filas comenzando por <SOS>.
        Convierte SMILES a SELFIES antes de tokenizar y los divide en tokens.
        """
        self.prompt_size = prompt_size
        self.prompts_input_ids = []

        for smiles in zinc_data:
            try:
                if pd.notna(smiles):  # Verificar que el SMILES no sea nulo
                    smiles = smiles.strip()
                    selfie = sf.encoder(smiles)  # Convertir SMILES a SELFIES
                    
                    # Verificar si la conversión fue exitosa
                    if selfie is None or selfie == "":
                        raise ValueError("SELFIES vacío después de la conversión")

                    # Dividir la cadena SELFIES en tokens
                    selfie_tokens = self.split_selfies(selfie)

                    # Tokenizar SELFIES
                    tokens = (
                        [stoi["<SOS>"]]
                        + POCKET_TOKENS
                        + [stoi["<LIGAND>"]]
                        + [stoi[token] for token in selfie_tokens if token in stoi]
                    )
                    # Descartar SELFIES demasiado cortos
                    if len(tokens) < prompt_size:
                        continue

                    # Truncar SELFIES largos
                    if len(tokens) > prompt_size:
                        tokens = tokens[:prompt_size]

                    # Agregar a la lista final
                    self.prompts_input_ids.append(tokens)
            except Exception as e:
                print(f"Error al convertir SMILES '{smiles}': {e}")
                continue  # Si hay error, omitir este SMILES

    def __getitem__(self, ix):
        return self.prompts_input_ids[ix]

    def __len__(self):
        return len(self.prompts_input_ids)

    @staticmethod
    def split_selfies(selfies: str):
        tokens = []
        token  = ""
        for char in selfies:
            token += char
            if char == "]":
                tokens.append(token)
                token = ""
        return tokens



    
class CustomPromptDataGenerator():
    def __init__(self, prompt_dataset, prompt_batch_size):
        self.prompt_dataset = prompt_dataset
        self.prompt_batch_size = prompt_batch_size

    def __iter__(self):
        self.dataset_indices = np.arange(len(self.prompt_dataset))
        return self

    def __next__(self):
        if len(self.dataset_indices) >= self.prompt_batch_size:
            # Seleccionar índices absolutos directamente desde self.dataset_indices
            picked_indices = np.random.choice(self.dataset_indices,
                                            self.prompt_batch_size,
                                            replace=False)
            # print(self.dataset_indices)  # Ver los índices actuales
            # print(picked_indices)       # Ver los índices seleccionados

            # Extraer las muestras correspondientes
            samples = [self.prompt_dataset[i] for i in picked_indices]

            # Actualizar self.dataset_indices eliminando los seleccionados
            self.dataset_indices = np.setdiff1d(self.dataset_indices, picked_indices, assume_unique=True)

            # Crear el batch
            input_ids = torch.tensor(samples, dtype=torch.long)
            batch = {'input_ids': input_ids}
            return batch
        else:
            raise StopIteration



# Si el archivo no existe, crea uno con el encabezado
if not os.path.exists(generated_smi_file):
    with open(generated_smi_file, 'w') as f:
        f.write("SMILES,Epoca,Recompensa\n")  # Encabezado con columnas adicionales

def calculate_logp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Descriptors.MolLogP(mol)
    else:
        return 0.0
    
def calculate_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Descriptors.MolWt(mol)
    else:
        return 0.0


def reward_fn(molecules_selfies, epoch):
    molecules_smiles = []

    for selfie in molecules_selfies:
        try:
            smiles = sf.decoder(selfie)  # SELFIES -> SMILES
            molecules_smiles.append(smiles)
        except Exception:
            molecules_smiles.append("C")

    valid_molecules = [m for m in molecules_smiles if m is not None]

    if len(valid_molecules) == 0:
        return [0.0] * len(molecules_selfies)

    temp_smiles_file = generated_smi_file
    docking_output_file_temp = os.path.join(os.path.dirname(temp_smiles_file), f"docking_results_2_{epoch}_temp.csv")
    global_docking_output_file = os.path.join(os.path.dirname(temp_smiles_file), f"docking_results_2_{epoch}.csv")

    with open(temp_smiles_file, 'w') as f:
        for mol in valid_molecules:
            f.write(mol + '\n')

    try:
        print(f"Llamando a {reward_script_2} para realizar el docking en la época {epoch}...")
        subprocess.run(["python", reward_script_2, temp_smiles_file, vars_file_2, str(epoch)], check=True)

        df_temp = pd.read_csv(docking_output_file_temp)

        if 'Docking' not in df_temp.columns:
            raise ValueError("El archivo de resultados no contiene la columna 'Docking'.")

        # Calcular logP y añadir columna
        df_temp['LogP'] = df_temp['SMILES'].apply(calculate_logp)


        # Calcular fitness combinando docking y logP
        df_temp['Fitness'] = df_temp.apply(lambda row: combine_docking_logP(row['Docking'], row['LogP']), axis=1)

        if os.path.exists(global_docking_output_file):
            df_global = pd.read_csv(global_docking_output_file)
            df_final = pd.concat([df_global, df_temp], ignore_index=True)
        else:
            df_final = df_temp

        df_final.to_csv(global_docking_output_file, index=False)
        fitness_scores = df_temp['Fitness'].tolist()

    except Exception as e:
        print(f"Error durante el cálculo de fitness: {e}")
        fitness_scores = [0.0] * len(valid_molecules)

    return fitness_scores



@dataclass
class PPORLElement:
    query_tensor: TensorType["query_size"]
    response_tensor: TensorType["response_size"]
    logprobs: TensorType["response_size"]
    values: TensorType["response_size"]
    rewards: TensorType["response_size"]


@dataclass
class PPORLBatch:
    query_tensors: TensorType["batch_size", "query_size"]
    response_tensors: TensorType["batch_size", "response_size"]
    logprobs: TensorType["batch_size", "response_size"]
    values: TensorType["batch_size", "response_size"]
    rewards: TensorType["batch_size", "response_size"]


class PPORolloutStorage():

    def __init__(self):
        super().__init__()
        self.pad_token_id = stoi ['<PAD>']
        self.history: Iterable[PPORLElement] = []

    def push(self, exps: Iterable[PPORLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def __getitem__(self, index: int) -> PPORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(self, mini_batch_size: int, shuffle: bool) -> DataLoader:
        def collate_fn(elems: Iterable[PPORLElement]):
            return PPORLBatch(
                pad_sequence(
                    [elem.query_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=float('nan'),
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.values for elem in elems],
                    padding_value=float('nan'),
                    batch_first=True
                ),
                pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=float('nan'),
                    batch_first=True,
                ),
            )

        return DataLoader(self, mini_batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
class Agent(nn.Module):
    def __init__(self, model_checkpoint_path, trainable=False, device='cuda', dtype=torch.bfloat16):
        """
        Clase para integrar PPO con tu modelo GPT, con una cabeza de valores adicional.
        """
        super().__init__()
        self.trainable = trainable
        self.device = device
        self.dtype = dtype
        self.generate_kwargs = dict(
            args.gen_kwargs,
            eos_token =stoi ['<EOS>'],
            pad_token =stoi ['<PAD>']
        )

        # Cargar el modelo desde el checkpoint
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        self.model = GPT(gptconf).to(device, dtype = dtype)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict, strict=False)

        # Configurar el modelo como entrenable o evaluado
        if not self.trainable:
            self.model.eval()
            self.model.requires_grad_(False)
        else:
            # Agregar una cabeza de valores si es entrenable
            self.model.train()
            self.model.requires_grad_(True)
            n_embd = self.model.config.n_embd  # Número de dimensiones del embedding
            num_labels = 1
            self.value_head = nn.Sequential(
                # nn.LayerNorm(n_embd),
                # nn.GELU(),
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, num_labels),
            ).to(dtype).to(device, dtype = dtype)

        # Obtener la capa de logits (similar a "logit_head")
        self.logit_head = self.model.lm_head
        self.block_size = self.model.config.block_size
        self.n_embd = self.model.config.n_embd

        pocket_np = pocket_emb_np.astype(np.float32)  # (NUM_RES, D_PROT)
        NUM_RES, D_PROT = pocket_np.shape
        assert D_PROT == self.n_embd, (
            f"Dim del pocket ({D_PROT}) != n_embd del modelo ({self.n_embd}). "
            "Si no coincide, habría que añadir una proyección lineal."
        )

        # matriz (T, n_embd) con ceros en toda la secuencia
        pocket_emb_full = np.zeros((self.block_size, self.n_embd), dtype=np.float32)

        # índice donde empiezan los AA en la secuencia: <SOS>(0), <POCKET>(1), AA1(2)...
        aa_start = 2
        L = min(NUM_RES, self.block_size - aa_start)

        # colocar los embeddings de ProtT5 exactamente en las posiciones de los AA
        pocket_emb_full[aa_start:aa_start + L, :] = pocket_np[:L, :]

        # guardar plantilla (T, n_embd) en el dispositivo del agente
        self.pocket_emb_template = torch.from_numpy(pocket_emb_full).to(
            self.device, dtype=self.dtype
        )

    def generate(self, input_ids, epoch):
        B, T = input_ids.shape
        pocket_emb = self.pocket_emb_template.unsqueeze(0).expand(B, -1, -1)
        return self.model.generate(
            idx=input_ids,  # Cambiar a `idx` para que coincida con `GPT.generate`
            pocket_emb=pocket_emb,
            **self.generate_kwargs, epoch = epoch
        )

    def forward(self, input_ids, attention_mask=None):
        # Llama al modelo subyacente con `return_hidden_states=True`
        B, T = input_ids.shape
        pocket_emb = self.pocket_emb_template.unsqueeze(0).expand(B, -1, -1)
        lm_logits, last_hidden_state = self.model(input_ids, attention_mask = attention_mask,pocket_emb=pocket_emb, return_hidden_states=True)
        
        if self.trainable:
            
            # Convertir last_hidden_state a torch.bfloat16 para que coincida con los pesos
            last_hidden_state = last_hidden_state.to(self.dtype)
            
            
            # Pasar por la value_head y calcular el valor
            value = self.value_head(last_hidden_state).squeeze(-1)
           

            return lm_logits, value
        else:
            return lm_logits



def logprobs_from_logits(logits, labels):
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)

class RolloutCreator():

    def __init__(
            self,
            prompt_dataset,
            prompt_batch_size=args.prompt_batch_size,
    ):
        self.prompt_batch_size = prompt_batch_size
        self.prompt_dataset = prompt_dataset
        self.prompt_generator = CustomPromptDataGenerator(self.prompt_dataset, self.prompt_batch_size)
        self.prompt_iterator = iter(self.prompt_generator)

        
    def make_experience(self, model, epoch, num_rollouts=128):
        all_rollouts = []
        while len(all_rollouts) < num_rollouts:
            try:
                batch = next(self.prompt_iterator)
            except StopIteration:
                self.prompt_generator = CustomPromptDataGenerator(self.prompt_dataset, self.prompt_batch_size)
                self.prompt_iterator = iter(self.prompt_generator)
                batch = next(self.prompt_iterator)

            # 1) Mover las entradas (prompts) a la GPU entrenable
            query_tensors = batch['input_ids'].to(train_device)

            # 2) Generar secuencias con el modelo entrenable
            trajectories = model.generate(
                query_tensors, epoch = epoch
            )
            print("Shape de trajectories:", trajectories.shape)  # 👈 aquí
            with open(f"trajectories_{epoch}.txt", "w") as f:
                for traj in trajectories:
                    token_ids = traj.tolist()
                    f.write(" ".join(map(str, token_ids)) + "\n")


            response_tensors = trajectories[:, query_tensors.shape[1]:]

            # 3) Crear la attention_mask en la misma GPU entrenable
            attention_mask = trajectories.eq(stoi["<PAD>"]).to(train_device)

            with torch.no_grad():
                # 4) Calcular logits y values con el modelo entrenable (train_device)
                logits, values = model(
                    trajectories, 
                    attention_mask=attention_mask,
                )

                # 5) Mover trajectories y attention_mask a ref_device para el modelo de referencia
                trajectories_ref = trajectories.to(ref_device)
                attention_mask_ref = attention_mask.to(ref_device)

                ref_logits = ref_model(
                    trajectories_ref,
                    attention_mask=attention_mask_ref,
                )

                # 6) Regresar ref_logits a la GPU entrenable (train_device)
                ref_logits = ref_logits.to(train_device)

            # ---- El resto sigue igual, ya que ahora logits y ref_logits están en train_device ----
            logprobs = logprobs_from_logits(logits[:, :-1, :], trajectories[:, 1:])
            ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], trajectories[:, 1:])
            n_trajectories = trajectories.shape[0]

            values = values[:, :-1]
            start = batch['input_ids'].shape[1] - 1
            valid_mask = ~attention_mask[:, 1:]            # ahora True donde hay tokens válidos
            valid_counts = valid_mask[:, start:].sum(1)    # nº tokens válidos a partir de 'start'
            ends = start + valid_counts                    # tensor con longitudes efectivas por secuencia

            truncated_values = [values[i, start : ends[i]].clone() for i in range(n_trajectories)]
            truncated_logprobs = [logprobs[i, start : ends[i]].clone() for i in range(n_trajectories)]

            truncated_values_padded   = pad_sequence(truncated_values,  batch_first=True, padding_value=float('nan'))
            truncated_logprobs_padded = pad_sequence(truncated_logprobs, batch_first=True, padding_value=float('nan'))


            # Decodificar, calcular reward, etc.
            texts = [decode(seq.tolist()) for seq in trajectories]
            texts = [strip_to_ligand(t) for t in texts]
            scores = reward_fn(texts, epoch)

            # KL por token + reward final
            rewards = -args.kl_coef * (logprobs - ref_logprobs)
            all_rewards = [rewards[i][start : ends[i]].clone() for i in range(n_trajectories)]
            for i in range(n_trajectories):
                all_rewards[i][-1] += scores[i]  # Reemplazar última recompensa por la real

            all_rewards_padded= pad_sequence(all_rewards,batch_first=True, padding_value=float('nan'))

            new_rollout = [
                PPORLElement(
                    query_tensor=query_tensors[i],
                    response_tensor=response_tensors[i],
                    logprobs=truncated_logprobs_padded[i],
                    values=truncated_values_padded[i],
                    rewards=all_rewards_padded[i],
                )
                for i in range(n_trajectories)
            ]
            all_rollouts += new_rollout

        score = torch.tensor(scores).mean().detach().cpu().item()
        return all_rollouts, score

    
def gae(values, rewards, mask):
    advantages = torch.zeros_like(rewards, device=rewards.device)
    last_advantage = 0
    last_value = 0

    with torch.no_grad():
        for t in reversed(range(rewards.shape[1])):  # Iterar en orden inverso
            delta = rewards[:, t] + args.gamma * last_value * mask[:, t] - values[:, t]
            last_advantage = delta + args.gamma * args.lam * last_advantage * mask[:, t]

            advantages[:, t] = last_advantage * mask[:, t]  # Ignorar posiciones con padding

            last_value = values[:, t] * mask[:, t]

        returns = advantages + values
    return advantages, returns
    

def ppo_loss(
    logprobs,
    values,
    old_logprobs,
    old_values,
    advantages,
    returns,
    mask,
):
    values_clipped = torch.clamp(
        values,
        old_values - args.cliprange_value,
        old_values + args.cliprange_value,
    )
    n = mask.sum()
    vf_loss1 = (values - returns) ** 2
    vf_loss2 = (values_clipped - returns) ** 2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
    pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n
    loss = pg_loss + args.vf_coef * vf_loss
    loss_value = loss.item()
    pg_loss_value = pg_loss.item()
    vf_loss_value = vf_loss.item()

    # Guardar pérdidas en un archivo de texto
    with open("loss_history_M.txt", "a") as f:
        f.write(f"{loss_value},{pg_loss_value},{vf_loss_value}\n")
    return loss

def loss_fn(mini_batch):
    query_tensors = mini_batch.query_tensors
    response_tensors = mini_batch.response_tensors
    old_logprobs = mini_batch.logprobs
    old_values = mini_batch.values
    old_rewards = mini_batch.rewards
    query_tensors   = query_tensors.to(train_device)
    response_tensors= response_tensors.to(train_device)
    old_logprobs    = old_logprobs.to(train_device)
    old_values      = old_values.to(train_device)
    old_rewards     = old_rewards.to(train_device)

    response_length = old_rewards.shape[1]
    mask = ~torch.isnan(old_rewards)
    mask = mask.float()
    old_values_masked = torch.where(mask == 1, old_values, torch.zeros_like(old_values))
    old_rewards_masked = torch.where(mask == 1, old_rewards, torch.zeros_like(old_rewards))
    old_logprobs_masked = torch.where(mask == 1, old_logprobs, torch.zeros_like(old_logprobs))
    advantages, returns = gae(old_values_masked, old_rewards_masked, mask)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    trajectories = torch.hstack([query_tensors, response_tensors])
    attention_mask = trajectories.eq(stoi["<PAD>"])  # bool, True=PAD
    logits, values_pred = model(trajectories, attention_mask=attention_mask)

    values_pred = values_pred[:, :-1]
    logprobs = logprobs_from_logits(logits[:, :-1, :], trajectories[:, 1:])
    attention_mask = attention_mask[:, :-1]

    start = query_tensors.shape[1] - 1
    end = start + response_length
    logprobs, values_pred = (
        logprobs[:, start:end],
        values_pred[:, start:end],
        # attention_mask[:, start:end],
    )
    loss = ppo_loss(
        logprobs=logprobs,
        values=values_pred,
        old_logprobs=old_logprobs_masked,
        old_values=old_values_masked,
        advantages=advantages,
        returns=returns,
        mask=mask,
    )
    
    return loss, old_rewards[:,-1].mean().item()

ref_model = Agent(model_checkpoint_path=ckpt_path, trainable=False, device=ref_device)
model = Agent(model_checkpoint_path=ckpt_path, trainable=True, device=train_device)

# for name, param in model.named_parameters():
#     print(f"Nombre de la capa: {name}, Forma: {param.shape}")

zinc_data = pd.read_csv(data_path)["smiles"].str.strip()
# zinc_data = pd.read_csv("data/very_big_molecules_selfies/full_dataset_EE_morfeus2.csv", sep=";")["Smiles"].str.strip()


prompt_dataset = PromptDataset(args.prompt_size, zinc_data)

store = PPORolloutStorage()
rollout_creator = RolloutCreator(prompt_dataset, prompt_batch_size=args.prompt_batch_size)

opt = torch.optim.AdamW(model.parameters(), args.lr)

total_steps = (args.num_rollouts//args.batch_size)*args.ppo_epochs*args.epochs
tbar = tqdm(initial=0, total=total_steps)
ckpt_ppo_path = os.path.join(out_dir, "ckpt_zink_chembl_cross_sequence_add_buche.pt")  # Ruta del checkpoint PPO

all_scores = []
best_score = -float("inf")  # Inicializamos con un valor muy bajo

for i in range(args.epochs):
    # Fase 1: Recolección de rollouts
    store.clear_history()
    rollouts, score = rollout_creator.make_experience(model, i, args.num_rollouts)
    store.push(rollouts)
    train_dataloader = store.create_loader(args.batch_size, shuffle=True)

    # Fase 2: Optimización
    for batch in train_dataloader:
        for _ in range(args.ppo_epochs):
            loss, reward = loss_fn(batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
            tbar.update()

    all_scores.append(score)
    print ('all_scores', all_scores)
    tbar.set_description(f"| Época: {i+1} | Recompensa Promedio: {score:.3f} |")
    # Guardar el modelo solo si mejora el score

    if score > best_score:
        best_score = score
        checkpoint_ppo = {
            "epoch": i + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "all_scores": all_scores,
        }
        torch.save(checkpoint_ppo, ckpt_ppo_path)
        print(f"✅ Nuevo mejor modelo guardado en {ckpt_ppo_path} (Época {i+1}, Recompensa: {best_score:.3f})")


