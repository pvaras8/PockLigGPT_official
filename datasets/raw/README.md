# Raw datasets

This folder should contain the original molecular datasets before preprocessing.

Expected structure:

- `chembl/`: filtered ChEMBL CSV files
- `crossdocked/`: original CrossDocked tables, including
  `merged_pocket_smiles.csv`
- `zinc20/`: ZINC dataset files
- `zinc250k/`: ZINC-250K CSV files used in RL/training workflows

These files are not included in the repository.

Users must:
- download them manually, or
- generate them using preprocessing pipelines

Example expected files:
- `chembl1_filtered.csv`
- `chembl2_filtered.csv`
- `zinc_250k.csv`
