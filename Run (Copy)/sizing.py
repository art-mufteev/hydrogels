#!/usr/bin/env python3
import math; from pathlib import Path; import yaml; from rdkit import Chem; from rdkit.Chem import AllChem
NA=6.02214076e23; M_WATER=18.015; M_NA=22.99; M_CL=35.45; R_KCAL=0.0019872041; T=298.15; N_CONF=20; PRUNE=0.5; HERE=Path(__file__).resolve().parent
def compute_Vw(smiles: str) -> float:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None: raise ValueError(f'MMFF has no parameters for {smiles}')
    params = AllChem.ETKDGv3(); params.randomSeed = 42; params.pruneRmsThresh = PRUNE
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=N_CONF, params=params))
    if not cids: raise ValueError(f'Embedding failed for {smiles}')
    volumes, energies = [], []
    for cid in cids:
        if AllChem.MMFFOptimizeMolecule(mol, confId=cid) == -1: continue
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        e = ff.CalcEnergy(); v = AllChem.ComputeMolVolume(mol, confId=cid)
        if math.isfinite(e) and math.isfinite(v) and v > 0: volumes.append(v); energies.append(e)
    if not volumes: raise ValueError(f'No usable conformers for {smiles}')
    kT = R_KCAL * T; e0 = min(energies)
    w = [math.exp(-(e - e0) / kT) for e in energies]
    wsum = sum(w)
    v_mean = sum(vi * wi for vi, wi in zip(volumes, w)) / wsum
    return v_mean * 1e-24 * NA
def compute_sizing(composition, monomers, cfg):
    M_avg = sum(f * monomers[n]['M'] for n, f in composition.items())
    V_w_avg = sum(f * compute_Vw(monomers[n]['smiles']) for n, f in composition.items())
    rho_pol = M_avg / (V_w_avg / cfg['K_pack'])
    V_cell = cfg['Lx'] * cfg['Ly'] * cfg['Lz'] * 1e-24
    phi_V = cfg['C0'] * M_avg / (1000.0 * cfg['Q'] * rho_pol)
    N = cfg['N']
    N_monomers_float = phi_V * V_cell * rho_pol / M_avg * NA
    N_chains = max(1, round(N_monomers_float / N))
    N_monomers = N_chains * N
    m_pol = N_monomers * M_avg / NA
    V_pol = m_pol / rho_pol
    V_liq = V_cell - V_pol
    if V_liq <= 0: raise ValueError(f'No liquid phase: V_cell={V_cell:.2e}, V_pol={V_pol:.2e}')
    V_liq_L = V_liq * 1e-3
    keys = list(composition.keys())
    cnt = [round(composition[s] * N) for s in keys]
    cnt[-1] = N - sum(cnt[:-1])
    nrepeat = dict(zip(keys, cnt))
    polymer_charge = N_chains * sum(nrepeat[n] * monomers[n]['charge'] for n in composition)
    c_net = polymer_charge / NA / V_liq_L
    C_out = cfg['C_out']
    C_na_in = (-c_net + math.sqrt(c_net**2 + 4 * C_out**2)) / 2.0
    C_cl_in = C_na_in + c_net
    N_Na = round(C_na_in * V_liq_L * NA)
    N_Cl = N_Na + polymer_charge
    if N_Cl < 0: raise ValueError(f'N_Cl={N_Cl} < 0; composition too anionic for this cell')
    m_water = V_liq * cfg['rho_solution'] - N_Na * M_NA / NA - N_Cl * M_CL / NA
    if m_water <= 0: raise ValueError('Water mass balance negative')
    N_water = round(m_water / M_WATER * NA)
    return {'M_avg': M_avg, 'V_w_avg': V_w_avg, 'rho_pol': rho_pol, 'phi_V': phi_V,
            'c_net': c_net, 'C_na_in': C_na_in, 'C_cl_in': C_cl_in,
            'polymer_charge': polymer_charge, 'nrepeat': nrepeat,
            'N_chains': N_chains, 'N_monomers': N_monomers,
            'N_water': N_water, 'N_Na': N_Na, 'N_Cl': N_Cl}
if __name__ == '__main__':
    monomers = yaml.safe_load((HERE / 'monomers.yaml').read_text())['monomers']
    composition = {'HEA': 14/101, 'BA': 59/101, 'CBEA': 8/101, 'ATAC': 10/101, 'PEA': 10/101}
    cfg = {'N': 100, 'C0': 2.4, 'Q': 0.42, 'C_out': 0.154, 'rho_solution': 1.004, 'K_pack': 0.681, 'Lx': 80.6296, 'Ly': 82.8640, 'Lz': 80.0}
    for k, v in compute_sizing(composition, monomers, cfg).items():
        print(f'{k:16s} = {v}')