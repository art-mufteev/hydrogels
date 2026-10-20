#!/usr/bin/env python3
import csv; from pathlib import Path; import yaml; from rdkit import Chem; from rdkit.Chem import Descriptors
SHORT_NAMES = {'C=CC(=O)OCCO':'HEA','C=CC(=O)OCCCC':'BA','C=CC(=O)OCCC(=O)[O-]':'CBEA','C=CC(=O)OCC[N+](C)(C)C':'ATAC','C=CC(=O)OCCOc1ccccc1':'PEA','C=CC(=O)N':'AAm','C=CC(=O)[O-]':'AA','C=C(C)C(=O)[O-]':'MAA','C=CC(=O)OC':'MA','C=CC(=O)OCC':'EA','C=C(C)C(=O)OC':'MMA','C=CC(=O)NC(C)C':'NIPAM','C=CC(=O)N(C)C':'DMA','C=CC(=O)OCCN(C)C':'DMAEA','C=CC1=CC=CC=C1':'Sty'}
MANUAL_EMC = {}
_CARBOXYL = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
def strip_counter_ions(smiles: str) -> str:
    keep = []
    for frag in smiles.split('.'):
        m = Chem.MolFromSmiles(frag)
        if m is not None and m.GetNumAtoms() == 1 and m.GetAtomWithIdx(0).GetSymbol() in {'Cl','Br','I','F','Na','K','Li','Cs','Rb'}: continue
        keep.append(frag)
    return '.'.join(keep) if keep else smiles
def deprotonate_carboxyls(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return smiles
    matches = mol.GetSubstructMatches(_CARBOXYL)
    if not matches: return Chem.MolToSmiles(mol)
    em = Chem.RWMol(mol)
    for _c, _o_double, o_oh in matches:
        atom = em.GetAtomWithIdx(o_oh)
        atom.SetFormalCharge(-1); atom.SetNumExplicitHs(0); atom.SetNoImplicit(True)
    return Chem.MolToSmiles(em.GetMol())
def compute_charge(mol) -> int: return sum(a.GetFormalCharge() for a in mol.GetAtoms())
def compute_M(mol) -> float: return round(Descriptors.MolWt(mol), 3)
def infer_emc_chemistry(smiles: str) -> str:
    if smiles in MANUAL_EMC: return MANUAL_EMC[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    patterns = [Chem.MolFromSmarts('[CH2]=[C](-[CH3])-[C](=O)-[#7,#8]'), Chem.MolFromSmarts('[CH2]=[CH]-[C](=O)-[#7,#8]')]
    for patt in patterns:
        match = mol.GetSubstructMatch(patt)
        if not match: continue
        ch2_idx, c_idx = match[0], match[1]
        em = Chem.RWMol(mol)
        d1 = em.AddAtom(Chem.Atom(0)); d2 = em.AddAtom(Chem.Atom(0))
        em.AddBond(ch2_idx, d1, Chem.BondType.SINGLE); em.AddBond(c_idx, d2, Chem.BondType.SINGLE)
        em.GetBondBetweenAtoms(ch2_idx, c_idx).SetBondType(Chem.BondType.SINGLE)
        return Chem.MolToSmiles(em.GetMol())
    return None
def process_row(row):
    original = row['smiles'].strip(); name = row['name'].strip(); number = int(row['number']); family = row['family'].strip()
    stripped = strip_counter_ions(original)
    normalized = deprotonate_carboxyls(stripped)
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(normalized))
    mol = Chem.MolFromSmiles(canonical)
    short = SHORT_NAMES.get(canonical) or SHORT_NAMES.get(stripped) or f'M{number:03d}'
    return {'short': short, 'name': name, 'family': family, 'smiles': canonical, 'M': compute_M(mol), 'charge': compute_charge(mol), 'emc_chemistry': infer_emc_chemistry(canonical)}
def main(csv_path: Path, out_path: Path):
    records = []
    for row in csv.DictReader(csv_path.open(encoding='utf-8')):
        try: records.append(process_row(row))
        except Exception as e: print(f"FAIL {row['name']}: {e}")
    seen = {}
    for r in records:
        s = r['short']; seen[s] = seen.get(s, 0) + 1
        if seen[s] > 1: r['short'] = f'{s}_{seen[s]}'
    missing = [r for r in records if r['emc_chemistry'] is None]
    if missing:
        print(f'\n{len(missing)} monomers need manual EMC chemistry:')
        for r in missing: print(f"  {r['short']:10s} | {r['smiles']}")
    out = {'monomers': {r['short']: {'name': r['name'], 'family': r['family'], 'smiles': r['smiles'], 'M': r['M'], 'charge': r['charge'], 'emc_chemistry': r['emc_chemistry']} for r in records}}
    out_path.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print(f'\nWrote {out_path} with {len(records)} monomers')
if __name__ == '__main__':
    import argparse; p = argparse.ArgumentParser(); p.add_argument('csv_path', type=Path); p.add_argument('--out', type=Path, default=Path('monomers.yaml')); args = p.parse_args(); main(args.csv_path, args.out)