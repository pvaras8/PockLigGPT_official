import json
from argparse import Namespace

from scripts.train_ppo import (
    _apply_cli_docking_overrides,
    _build_experiment_layout,
    _load_docking_vars,
)


def test_run_uses_a_private_docking_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("RECEPTOR\n", encoding="utf-8")

    template = tmp_path / "vars_meeko.json"
    original = {
        "size_x": 20.0,
        "size_y": 20.0,
        "size_z": 20.0,
    }
    template.write_text(json.dumps(original), encoding="utf-8")

    cfg = {
        "reward": {
            "docking": {
                "vars_file": str(template),
                "script": "scripts/reward_meeko_vina.py",
            }
        },
        "output": {
            "ppo_ckpt_path": "ppo_best.pt",
            "loss_history_file": "loss_history.csv",
        },
    }
    args = Namespace(
        pdbqt_path=str(receptor),
        center="1.0,2.0,3.0",
    )

    vars_cfg = _load_docking_vars(cfg)
    _apply_cli_docking_overrides(vars_cfg, args)
    _build_experiment_layout(cfg, vars_cfg)

    assert json.loads(template.read_text(encoding="utf-8")) == original

    run_vars = cfg["reward"]["docking"]["vars_file"]
    with open(run_vars, encoding="utf-8") as handle:
        run_cfg = json.load(handle)

    assert run_cfg["filename_of_receptor"] == str(receptor)
    assert run_cfg["center_x"] == 1.0
    assert run_cfg["center_y"] == 2.0
    assert run_cfg["center_z"] == 3.0
