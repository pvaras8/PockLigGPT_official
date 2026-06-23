import json
import os
import sys
import contextlib
import shlex
import subprocess
from multiprocessing import cpu_count, get_context
from pathlib import Path
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


def mol_to_mol2_file(
    mol,
    mol2_path: str,
    obabel_bin: str,
    conda_env: str = "",
    activation: str = "conda_run",
    conda_bin: str = "conda",
) -> None:
    sdf_path = os.path.splitext(mol2_path)[0] + ".sdf"
    Chem.MolToMolFile(mol, sdf_path)

    if conda_env:
        obabel_cmd = (
            f"{shlex.quote(obabel_bin)} -isdf {shlex.quote(sdf_path)} "
            f"-omol2 -O {shlex.quote(mol2_path)}"
        )
        if activation == "source_activate":
            cmd = f"source activate {shlex.quote(conda_env)} && {obabel_cmd}"
        else:
            cmd = (
                f"{shlex.quote(conda_bin)} run -n {shlex.quote(conda_env)} "
                f"{obabel_cmd}"
            )
        completed = subprocess.run(
            ["bash", "-lc", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    else:
        completed = subprocess.run(
            [obabel_bin, "-isdf", sdf_path, "-omol2", "-O", mol2_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    if completed.returncode != 0 or not os.path.exists(mol2_path):
        raise RuntimeError(
            "OpenBabel no pudo generar mol2: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def _affinity_cfg(cfg: dict) -> dict:
    affinity = dict(cfg.get("affinity", {}))
    affinity["enabled"] = bool(affinity.get("enabled", False))
    affinity["model"] = str(affinity.get("model", "svr_fp_pose"))
    affinity["conda_env"] = str(affinity.get("conda_env", "pockliggpt_affinity_svr"))
    affinity["project_dir"] = os.path.abspath(
        affinity.get("project_dir", "Affinity_Prediction_Model")
    )
    affinity["script"] = os.path.abspath(
        affinity.get(
            "script",
            os.path.join(affinity["project_dir"], "predict_affinity.py"),
        )
    )
    affinity["score_column"] = str(
        affinity.get("score_column", "predicted_affinity_kcal_mol")
    )
    affinity["selection"] = str(affinity.get("selection", "min"))
    affinity["obabel_bin"] = str(affinity.get("obabel_bin", "obabel"))
    affinity["fallback_to_vina"] = bool(affinity.get("fallback_to_vina", False))
    affinity["device"] = str(affinity.get("device", "cpu"))
    affinity["python_bin"] = str(affinity.get("python_bin", "python"))
    affinity["conda_bin"] = str(affinity.get("conda_bin", "conda"))
    affinity["activation"] = str(affinity.get("activation", "conda_run"))
    return affinity


def _validate_affinity_cfg(affinity: dict) -> None:
    if not affinity["enabled"]:
        return
    if affinity["model"] not in {"svr_fp_pose", "dimenet"}:
        raise ValueError("affinity.model debe ser 'svr_fp_pose' o 'dimenet'")
    if affinity["selection"] not in {"min", "max", "first"}:
        raise ValueError("affinity.selection debe ser 'min', 'max' o 'first'")
    if affinity["activation"] not in {"conda_run", "source_activate"}:
        raise ValueError(
            "affinity.activation debe ser 'conda_run' o 'source_activate'"
        )
    if not os.path.exists(affinity["script"]):
        raise FileNotFoundError(
            f"No se encontró predict_affinity.py: {affinity['script']}"
        )
    if affinity["model"] == "dimenet" and not affinity.get("protein_pdb"):
        raise ValueError("affinity.protein_pdb es obligatorio con model='dimenet'")


def _maybe_abs(path_value: str, base_dir: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(Path(base_dir) / path)


def _build_affinity_command(affinity: dict, manifest_path: str, output_path: str) -> str:
    parts = [
        shlex.quote(affinity["python_bin"]),
        shlex.quote(affinity["script"]),
        "--manifest",
        shlex.quote(manifest_path),
        "--model",
        shlex.quote(affinity["model"]),
        "--output",
        shlex.quote(output_path),
    ]

    project_dir = affinity["project_dir"]
    if affinity["model"] == "svr_fp_pose":
        svr_model = _maybe_abs(
            affinity.get("svr_model", "checkpoints/svr_affinity_fp_pose_model.joblib"),
            project_dir,
        )
        svr_meta = _maybe_abs(
            affinity.get("svr_meta", "checkpoints/svr_affinity_fp_pose_metadata.json"),
            project_dir,
        )
        parts.extend(["--svr-model", shlex.quote(svr_model)])
        parts.extend(["--svr-meta", shlex.quote(svr_meta)])
    else:
        dimenet_model = _maybe_abs(
            affinity.get("dimenet_model", "checkpoints/dimenet_model.pt"),
            project_dir,
        )
        parts.extend(["--dimenet-model", shlex.quote(dimenet_model)])
        parts.extend(["--device", shlex.quote(affinity["device"])])

    cmd = " ".join(parts)
    if affinity.get("conda_env"):
        if affinity["activation"] == "source_activate":
            cmd = f"source activate {shlex.quote(affinity['conda_env'])} && {cmd}"
        else:
            cmd = (
                f"{shlex.quote(affinity['conda_bin'])} run "
                f"-n {shlex.quote(affinity['conda_env'])} {cmd}"
            )
    return cmd


def _run_affinity_prediction(
    df: pd.DataFrame,
    affinity: dict,
    final_folder: str,
    epoch: str,
    fallback_score: float,
) -> pd.DataFrame:
    if not affinity["enabled"]:
        return df

    ok_mask = (
        (df["Vina_status"] == "OK")
        & df["Vina_ligand_mol2"].notna()
        & df["Vina_pose_pdbqt"].notna()
    )
    manifest_df = df.loc[ok_mask, ["ID", "Vina_ligand_mol2", "Vina_pose_pdbqt"]].copy()
    manifest_df = manifest_df.rename(
        columns={
            "ID": "ligand_id",
            "Vina_ligand_mol2": "ligand_mol2",
            "Vina_pose_pdbqt": "vina_pdbqt",
        }
    )
    protein_pdb = affinity.get("protein_pdb", "")
    manifest_df["protein_pdb"] = os.path.abspath(protein_pdb) if protein_pdb else ""

    manifest_path = os.path.join(final_folder, f"affinity_manifest_{epoch}.csv")
    prediction_path = os.path.join(final_folder, f"affinity_predictions_{epoch}.csv")
    manifest_df.to_csv(manifest_path, index=False)

    if manifest_df.empty:
        df["Affinity_status"] = "SKIPPED_NO_VALID_VINA"
        df["Affinity_prediction"] = fallback_score
        if not affinity["fallback_to_vina"]:
            df["Docking"] = fallback_score
        return df

    completed = subprocess.run(
        ["bash", "-lc", _build_affinity_command(affinity, manifest_path, prediction_path)],
        cwd=affinity["project_dir"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        if affinity["fallback_to_vina"]:
            df["Affinity_status"] = "ERROR_FALLBACK_TO_VINA"
            df["Affinity_error"] = completed.stderr.strip() or completed.stdout.strip()
            return df
        raise RuntimeError(
            "Affinity prediction falló:\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    if not os.path.exists(prediction_path):
        raise FileNotFoundError(
            f"predict_affinity.py no generó el CSV esperado: {prediction_path}"
        )

    pred_df = pd.read_csv(prediction_path)
    score_col = affinity["score_column"]
    if score_col not in pred_df.columns:
        raise ValueError(
            f"El CSV de affinity no contiene '{score_col}'. "
            f"Columnas disponibles: {list(pred_df.columns)}"
        )

    if pred_df.empty:
        df["Affinity_status"] = "NO_PREDICTIONS"
        df["Affinity_prediction"] = fallback_score
        if not affinity["fallback_to_vina"]:
            df.loc[df["Vina_status"] == "OK", "Docking"] = fallback_score
        return df

    pred_df[score_col] = pd.to_numeric(pred_df[score_col], errors="coerce")
    pred_df = pred_df.dropna(subset=[score_col])
    if pred_df.empty:
        df["Affinity_status"] = "NO_NUMERIC_PREDICTIONS"
        df["Affinity_prediction"] = fallback_score
        if not affinity["fallback_to_vina"]:
            df.loc[df["Vina_status"] == "OK", "Docking"] = fallback_score
        return df

    if affinity["selection"] == "min":
        best_df = pred_df.loc[pred_df.groupby("ligand_id")[score_col].idxmin()]
    elif affinity["selection"] == "max":
        best_df = pred_df.loc[pred_df.groupby("ligand_id")[score_col].idxmax()]
    else:
        best_df = (
            pred_df.sort_values(["ligand_id", "pose_idx"])
            .groupby("ligand_id")
            .head(1)
        )

    score_by_id = {
        str(row.ligand_id): float(getattr(row, score_col))
        for row in best_df.itertuples(index=False)
    }

    df["Affinity_status"] = "SKIPPED"
    df["Affinity_prediction"] = fallback_score
    for row_idx, row in df.iterrows():
        ligand_id = str(row["ID"])
        if ligand_id in score_by_id:
            df.at[row_idx, "Docking"] = score_by_id[ligand_id]
            df.at[row_idx, "Affinity_prediction"] = score_by_id[ligand_id]
            df.at[row_idx, "Affinity_status"] = "OK"
        elif row["Vina_status"] == "OK":
            df.at[row_idx, "Affinity_status"] = "NO_PREDICTION"
            if not affinity["fallback_to_vina"]:
                df.at[row_idx, "Docking"] = fallback_score
                df.at[row_idx, "Affinity_prediction"] = fallback_score

    df["Affinity_manifest"] = manifest_path
    df["Affinity_predictions_csv"] = prediction_path
    return df


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
    ligand_mol2_path = os.path.join(WORKER_STATE["ligands_mol2_folder"], f"{mol_id}.mol2")
    pose_pdbqt_path = os.path.join(WORKER_STATE["poses_folder"], f"{mol_id}_out.pdbqt")

    try:
        mol = smiles_to_3d_mol(smiles, seed=int(WORKER_STATE["embed_seed"]) + idx)
        if WORKER_STATE["affinity_enabled"]:
            mol_to_mol2_file(
                mol,
                ligand_mol2_path,
                WORKER_STATE["obabel_bin"],
                WORKER_STATE["obabel_conda_env"],
                WORKER_STATE["conda_activation"],
                WORKER_STATE["conda_bin"],
            )
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
            "Vina_ligand_mol2": ligand_mol2_path,
            "Vina_pose_pdbqt": pose_pdbqt_path,
        }

    except Exception as exc:
        return idx, {
            "ID": str(idx + 1),
            "SMILES": smiles,
            "Docking": WORKER_STATE["fallback_score"],
            "Vina_status": "ERROR",
            "Vina_ligand_pdbqt": ligand_pdbqt_path,
            "Vina_ligand_mol2": ligand_mol2_path,
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
    affinity = _affinity_cfg(cfg)
    _validate_affinity_cfg(affinity)

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
    ligands_mol2_folder = os.path.join(final_folder, f"ligands_mol2_epoch_{epoch}")
    os.makedirs(poses_folder, exist_ok=True)
    os.makedirs(ligands_folder, exist_ok=True)
    if affinity["enabled"]:
        os.makedirs(ligands_mol2_folder, exist_ok=True)

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
        "ligands_mol2_folder": ligands_mol2_folder,
        "affinity_enabled": affinity["enabled"],
        "obabel_bin": affinity["obabel_bin"],
        "obabel_conda_env": affinity.get("obabel_conda_env", affinity["conda_env"]),
        "conda_activation": affinity["activation"],
        "conda_bin": affinity["conda_bin"],
    }

    tasks = []
    results = [None] * len(smiles_list)
    fallback_score = float(cfg.get("fallback_score", -6.0))

    for idx, smiles in enumerate(smiles_list):
        mol_id = f"mol_{idx + 1:06d}"
        ligand_pdbqt_path = os.path.join(ligands_folder, f"{mol_id}_input.pdbqt")
        ligand_mol2_path = os.path.join(ligands_mol2_folder, f"{mol_id}.mol2")
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
                "Vina_ligand_mol2": ligand_mol2_path,
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
        if affinity["enabled"]:
            print("---------- 2. Affinity prediction ----------")
            print(f"Modelo: {affinity['model']}")
            print(f"Conda env: {affinity['conda_env']}")
            print(f"Ligands mol2: {ligands_mol2_folder}")

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
    df = _run_affinity_prediction(
        df=df,
        affinity=affinity,
        final_folder=final_folder,
        epoch=epoch,
        fallback_score=fallback_score,
    )
    df.to_csv(output_csv_path, index=False)

    if verbose:
        print(f"Resultados guardados en: {output_csv_path}")


if __name__ == "__main__":
    main()
