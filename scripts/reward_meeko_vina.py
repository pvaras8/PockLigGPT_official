import json
import os
import sys
import contextlib
from multiprocessing import cpu_count, get_context
from typing import List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina


# ============================================================
# Utilidades
# ============================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = json.load(f)

    required = [
        "filename_of_receptor",
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
        "final_folder",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Faltan claves en vars.json: {missing}")

    return cfg


def read_smiles_file(smiles_path: str) -> List[str]:
    smiles = []
    with open(smiles_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            smi = line.split()[0]
            smiles.append(smi)
    return smiles


def smiles_to_3d_mol(smiles: str, seed: int = 42):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES inválido")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError("No se pudo generar conformero 3D")

    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is not None:
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass

    return mol


def mol_to_pdbqt_string(mol) -> str:
    preparator = MoleculePreparation()
    setups = preparator.prepare(mol)

    if not setups or len(setups) == 0:
        raise RuntimeError("Meeko no devolvió MoleculeSetup")

    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setups[0])

    if not is_ok:
        raise RuntimeError(f"Error al escribir PDBQT con Meeko: {error_msg}")

    return pdbqt_string


# ============================================================
# Estado global por worker
# ============================================================

WORKER_STATE = {}


def _resolve_worker_count(num_processors: int) -> int:
    if num_processors == -1:
        return max(1, cpu_count() - 1)
    return max(1, num_processors)


def _resolve_safe_parallelism(num_processors: int, vina_cpu_per_job: int) -> int:
    requested_workers = _resolve_worker_count(num_processors)
    cpus_per_job = max(1, int(vina_cpu_per_job))
    available = max(1, cpu_count())
    max_workers_by_cpu = max(1, available // cpus_per_job)
    return max(1, min(requested_workers, max_workers_by_cpu))


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Silence C/C++-level noisy output from Vina/RDKit during heavy worker steps."""
    with open(os.devnull, "w") as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def init_worker(cfg: dict):
    global WORKER_STATE
    WORKER_STATE = cfg


def worker(task: Tuple[int, str]):
    idx, smiles = task
    mol_id = f"mol_{idx + 1:06d}"
    ligand_pdbqt_path = os.path.join(WORKER_STATE["ligands_folder"], f"{mol_id}_input.pdbqt")
    pose_pdbqt_path = os.path.join(WORKER_STATE["poses_folder"], f"{mol_id}_out.pdbqt")

    try:
        mol = smiles_to_3d_mol(smiles, seed=int(WORKER_STATE["embed_seed"]) + idx)
        ligand_pdbqt = mol_to_pdbqt_string(mol)
        with open(ligand_pdbqt_path, "w") as f:
            f.write(ligand_pdbqt)

        with suppress_stdout_stderr():
            v = Vina(
                sf_name=WORKER_STATE["sf_name"],
                cpu=int(WORKER_STATE["vina_cpu_per_job"]),
                seed=int(WORKER_STATE["vina_seed"]) + idx,
                verbosity=0,
            )
            v.set_receptor(WORKER_STATE["filename_of_receptor"])
            v.compute_vina_maps(
                center=WORKER_STATE["center"],
                box_size=WORKER_STATE["box_size"],
            )
            v.set_ligand_from_string(ligand_pdbqt)
            v.dock(
                exhaustiveness=WORKER_STATE["exhaustiveness"],
                n_poses=WORKER_STATE["n_poses"],
            )

        energies = v.energies(
            n_poses=WORKER_STATE["n_poses"],
            energy_range=WORKER_STATE["energy_range"],
        )
        if energies is None or len(energies) == 0:
            raise RuntimeError("Vina no devolvió energías")

        score = float(energies[0][0])

        v.write_poses(
            pose_pdbqt_path,
            n_poses=WORKER_STATE["write_n_poses"],
            energy_range=WORKER_STATE["energy_range"],
            overwrite=True,
        )

        return idx, {
            "ID": str(idx + 1),
            "SMILES": smiles,
            "Docking": score,
            "Vina_status": "OK",
            "Vina_ligand_pdbqt": ligand_pdbqt_path,
            "Vina_pose_pdbqt": pose_pdbqt_path,
        }

    except Exception as exc:
        return idx, {
            "ID": str(idx + 1),
            "SMILES": smiles,
            "Docking": WORKER_STATE["fallback_score"],
            "Vina_status": "ERROR",
            "Vina_ligand_pdbqt": ligand_pdbqt_path,
            "Vina_pose_pdbqt": pose_pdbqt_path,
            "Vina_error": str(exc),
        }


# ============================================================
# Main
# ============================================================

def main():
    # Hide noisy RDKit messages (e.g., UFFTYPER warnings) during massive docking batches.
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")

    if len(sys.argv) != 4:
        raise SystemExit(
            "Uso: python reward_vina.py <source_compound_file> <vars.json> <epoch>"
        )

    source_compound_file = os.path.abspath(sys.argv[1])
    vars_json = os.path.abspath(sys.argv[2])
    epoch = str(sys.argv[3])

    cfg = load_config(vars_json)
    verbose = bool(cfg.get("verbose", False))

    if not os.path.exists(source_compound_file):
        raise FileNotFoundError(f"No se encontró el archivo de SMILES: {source_compound_file}")

    receptor_pdbqt = os.path.abspath(cfg["filename_of_receptor"])
    if not receptor_pdbqt.lower().endswith(".pdbqt"):
        raise ValueError(
            "El receptor debe ser un archivo .pdbqt (requerido por Meeko/Vina)"
        )
    if not os.path.exists(receptor_pdbqt):
        raise FileNotFoundError(f"No se encontró el receptor PDBQT: {receptor_pdbqt}")

    smiles_list = read_smiles_file(source_compound_file)
    if len(smiles_list) == 0:
        raise ValueError("El archivo de entrada no contiene SMILES")

    center = [
        float(cfg["center_x"]),
        float(cfg["center_y"]),
        float(cfg["center_z"]),
    ]
    box_size = [
        float(cfg["size_x"]),
        float(cfg["size_y"]),
        float(cfg["size_z"]),
    ]

    vina_cpu_per_job = int(cfg.get("vina_cpu_per_job", 1))
    vina_cpu_per_job = max(1, vina_cpu_per_job)
    n_workers = _resolve_safe_parallelism(int(cfg.get("num_processors", -1)), vina_cpu_per_job)
    max_mw = float(cfg.get("max_mw", 500.0))

    final_folder = os.path.abspath(cfg["final_folder"])
    os.makedirs(final_folder, exist_ok=True)

    output_csv_path = os.path.join(
        final_folder, f"docking_results_{epoch}_temp.csv"
    )

    poses_folder = os.path.join(final_folder, f"poses_epoch_{epoch}")
    ligands_folder = os.path.join(final_folder, f"ligands_epoch_{epoch}")
    os.makedirs(poses_folder, exist_ok=True)
    os.makedirs(ligands_folder, exist_ok=True)

    worker_cfg = {
        "filename_of_receptor": receptor_pdbqt,
        "center": center,
        "box_size": box_size,
        "exhaustiveness": int(cfg.get("exhaustiveness", 8)),
        "n_poses": int(cfg.get("n_poses", 1)),
        "write_n_poses": int(cfg.get("write_n_poses", int(cfg.get("n_poses", 1)))),
        "energy_range": float(cfg.get("energy_range", 3.0)),
        "vina_cpu_per_job": vina_cpu_per_job,
        "fallback_score": float(cfg.get("fallback_score", -6.0)),
        "sf_name": str(cfg.get("sf_name", "vina")),
        "embed_seed": int(cfg.get("embed_seed", 42)),
        "vina_seed": int(cfg.get("vina_seed", 12345)),
        "poses_folder": poses_folder,
        "ligands_folder": ligands_folder,
    }

    tasks = []
    results = [None] * len(smiles_list)
    fallback_score = float(cfg.get("fallback_score", -6.0))

    for idx, smiles in enumerate(smiles_list):
        mol_id = f"mol_{idx + 1:06d}"
        ligand_pdbqt_path = os.path.join(ligands_folder, f"{mol_id}_input.pdbqt")
        pose_pdbqt_path = os.path.join(poses_folder, f"{mol_id}_out.pdbqt")

        try:
            mol = Chem.MolFromSmiles(smiles)
            mw = float(Descriptors.MolWt(mol)) if mol is not None else None
        except Exception:
            mw = None

        if mw is not None and mw > max_mw:
            results[idx] = {
                "ID": str(idx + 1),
                "SMILES": smiles,
                "Docking": fallback_score,
                "Vina_status": "FILTERED_MW",
                "Vina_ligand_pdbqt": ligand_pdbqt_path,
                "Vina_pose_pdbqt": pose_pdbqt_path,
                "MW": mw,
            }
            continue

        tasks.append((idx, smiles))

    if verbose:
        print("---------- 1. Prediccion Docking ----------")
        print(f"Moléculas: {len(smiles_list)}")
        print(f"Workers: {n_workers}")
        print(f"Vina CPU/job: {vina_cpu_per_job}")
        print(f"Receptor: {receptor_pdbqt}")
        print(f"Centro: {center}")
        print(f"Box: {box_size}")

    if len(tasks) > 0:
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(worker_cfg,),
        ) as pool:
            done = 0
            for idx, res in pool.imap_unordered(worker, tasks):
                results[idx] = res
                done += 1
                if verbose:
                    print(f"[{done}/{len(tasks)}] OK")

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)

    if verbose:
        print(f"Resultados guardados en: {output_csv_path}")


if __name__ == "__main__":
    main()