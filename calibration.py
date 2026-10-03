"""Single-anchor regional bias correction.

Anchor = ONE real wet-lab measurement (USER_0001). The base v4 models are NOT
retrained or altered; this is a post-hoc correction with a smooth Gaussian
falloff, so predictions far from the anchor are unaffected. Any output that
received a non-zero correction is flagged, both here and in the UI, so a
calibrated number is never mistaken for an independent prediction.
"""
import numpy as np

ANCHOR_RAW = {'ee_percent': 99.46, 'dl_percent': 4.74, 'size_drug_nm': 134.6}
ANCHOR_ROW = {'fratio': 0.44, 'fphil': 2000.0, 'fphob': 872.1, 'inter1': 872.1 / 2001.0,
              'inter2': 0.44 * np.log1p(872.1), 'MolWt': 543.52, 'LogP': 1.27, 'TPSA': 206.0,
              'delta_drug': 10.185033782892889, 'delta_core': 19.2, 'chi_dc': 3.618561791604014,
              'chi_cw': 33.33797068692911, 'Kam': 1.546073825910337e-09,
              'preparation_method': 'film_hydration', 'fam': 'PCL', 'data_source': 'film_hydration'}
CENTER_CHI = ANCHOR_ROW['chi_dc']
CENTER_LOGMW = float(np.log(ANCHOR_ROW['fphob']))
SIGMA_CHI = 1.0
SIGMA_LOGMW = 1.2
MIN_WEIGHT = 0.05

def _weight(row):
    try:
        chi = float(row.get('chi_dc', 0.0)); mw = float(row.get('fphob', 0.0))
    except (TypeError, ValueError):
        return 0.0
    if mw <= 0:
        return 0.0
    w = np.exp(-0.5 * ((chi - CENTER_CHI) / SIGMA_CHI) ** 2
               - 0.5 * ((np.log(mw) - CENTER_LOGMW) / SIGMA_LOGMW) ** 2)
    return float(w) if w >= MIN_WEIGHT else 0.0

def _space(raw, target):
    """predict.py/UI work in raw units; Size is corrected in log1p space."""
    return float(np.log1p(max(raw, -0.99))) if target == 'Size' else float(raw)

def _unspace(v, target):
    return float(np.expm1(v)) if target == 'Size' else float(v)

ANCHOR_KEY = {'EE': 'ee_percent', 'DL': 'dl_percent', 'Size': 'size_drug_nm'}

def residual(anchor_pred_raw, target):
    return _space(ANCHOR_RAW[ANCHOR_KEY[target]], target) - _space(anchor_pred_raw, target)

def calibrate(target, raw_pred, row, anchor_pred_raw):
    if target not in ANCHOR_KEY:
        return raw_pred, 0.0
    w = _weight(row)
    if w <= 0:
        return raw_pred, 0.0
    r = residual(anchor_pred_raw, target)
    out = _unspace(_space(raw_pred, target) + w * r, target)
    if target == 'Size':
        out = max(out, 10.0)
    else:
        out = float(np.clip(out, 0.0, 100.0))
    return round(out, 3), round(w, 3)
