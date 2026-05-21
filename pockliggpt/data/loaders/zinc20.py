import os

def load_zinc_smiles(files):

    smiles = []

    for path in files:
        if not os.path.exists(path):
            continue

        with open(path) as f:
            for line in f:
                cols = line.strip().split()
                if cols and cols[0].lower() != "smiles":
                    smiles.append(cols[0])

    return smiles