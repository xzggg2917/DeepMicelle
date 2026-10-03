"""DeepMicelle prediction API: v6 models (EE/DL/Size) + v4 logCMC; calibration OFF by default now that v6 learned the region."""
import pickle, json, pandas as pd, numpy as np
import calibration as CAL

Q = json.load(open('reports/v6_metrics.json'))
_CACHE = {}
_V6 = {'EE', 'DL', 'Size'}

def _bundle(target):
    if target not in _CACHE:
        tag = 'v6' if target in _V6 else 'v4'
        _CACHE[target] = pickle.load(open(f'models/{tag}_{target}.pkl', 'rb'))
    return _CACHE[target]

def _raw_predict(target, row):
    b = _bundle(target)
    X = pd.DataFrame([row])
    for c in b['num']:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    for c in b['cat']:
        X[c] = X[c].astype(str)
    v = float(b['model'].predict(b['pre'].transform(X[b['num'] + b['cat']]))[0])
    return float(np.expm1(v)) if b['log1p'] else v

def predict(target: str, row: dict, use_calibration=False):
    raw = _raw_predict(target, row)
    value, w = (CAL.calibrate(target, raw, row, _raw_predict(target, CAL.ANCHOR_ROW))
                if use_calibration else (raw, 0.0))
    q = Q[target]['q90']
    flags = []
    if row.get('chi_dc', 0) > 2:
        flags.append('chi_dc>2: drug/core likely incompatible, verify experimentally')
    if str(row.get('fam', 'UNK')) == 'UNK':
        flags.append('new carrier family')
    if w > 0:
        flags.append(f'anchor-calibrated (w={w}), based on 1 wet-lab point')
    span = q if not flags or w > 0 else 1.5 * q
    lo, hi = value - span, value + span
    if target == 'Size':
        lo = max(lo, 0.0)
    if target in ('EE', 'DL'):
        lo, hi = max(lo, 0.0), min(hi, 100.0)
    return {'pred': round(value, 3), 'pred_raw': round(raw, 3), 'interval_90': [round(lo, 3), round(hi, 3)],
            'MAE_cv': Q[target]['MAE'], 'calibration_weight': w, 'flags': flags}

if __name__ == '__main__':
    demo = {'fratio': 0.44, 'fphil': 2000.0, 'fphob': 872.1, 'inter1': 872.1 / 2001.0,
            'inter2': 0.44 * np.log1p(872.1), 'MolWt': 543.52, 'LogP': 1.27, 'TPSA': 206.0,
            'delta_drug': 10.185, 'delta_core': 19.2, 'chi_dc': 3.6186, 'chi_cw': 33.338, 'Kam': 1.5e-09,
            'preparation_method': 'film_hydration', 'fam': 'PCL', 'data_source': 'film_hydration'}
    for t in ['EE', 'DL', 'Size', 'logCMC']:
        print(t, predict(t, demo))
