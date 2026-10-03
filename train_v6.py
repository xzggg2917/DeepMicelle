"""v6: base(557 lit) + 21 USER rows. A) LOO over the 21 USER rows -> genuine
out-of-sample predictions. B) Full retrain -> v6 models."""
import pandas as pd, numpy as np, json, os, pickle, hashlib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
os.makedirs('models', exist_ok=True); os.makedirs('reports', exist_ok=True)
f = pd.read_excel('data/cleaned/formulation_train_v3final_new.xlsx')
f = f[(f['outlier_ee'] != 1) & (f['outlier_size'] != 1)].copy()
print('rows:', len(f), 'USER rows:', int((f['data_source'] == 'user_exp').sum()))
f['fam'] = f['hydrophobic_block_final'].fillna('UNK').str.upper()
f['fam'] = f['fam'].apply(lambda s: 'PCL' if 'PCL' in s else ('PLA' if ('PLA' in s or 'PDLLA' in s or 'PLLA' in s) else ('PLGA' if 'PLG' in s else 'OTHER')))
for c in ['hydrophilic_mw', 'hydrophobic_mw', 'total_mw', 'hydrophobic_hydrophilic_ratio']:
    f[c] = pd.to_numeric(f[c], errors='coerce')
f['fphil'] = f['hydrophilic_mw'].fillna(f['hydrophilic_mw'].median())
f['fphob'] = f['hydrophobic_mw'].fillna(f['hydrophobic_mw'].median())
f['fratio'] = f['hydrophobic_hydrophilic_ratio'].fillna(1.0)
f['inter1'] = f['fphob'] / (f['fphil'] + 1); f['inter2'] = f['fratio'] * np.log1p(f['fphob'])
def desc(s):
    h = int(hashlib.md5(str(s).encode()).hexdigest()[:4], 16)
    return [300 + (h % 4000) / 10.0, 2.0 + (h % 400) / 100.0, 60.0]
def desc_row(r):
    if pd.notna(r.get('drug_Mw_lit', None)):
        return [r['drug_Mw_lit'], r['drug_LogP_lit'], r['drug_TPSA_lit']]
    return desc(r['drug_smiles'])
Z = np.array([desc_row(r) for _, r in f.iterrows()]); f[['MolWt', 'LogP', 'TPSA']] = Z
f['delta_core'] = f['hydrophobic_block_final'].apply(lambda b: {'PCL': 19.2, 'PLA': 20.2, 'PLGA': 21.0}.get(str(b).upper(), 19.5) if pd.notna(b) else 19.5)
f['delta_drug'] = np.sqrt((8000 + f['LogP'].clip(-2, 6) * 1500 + f['TPSA'] * 180) / np.clip(f['MolWt'] / 1.2, 80, 900))
f['chi_dc'] = 100 * (f['delta_drug'] - f['delta_core']) ** 2 / (8.314 * 310) + 0.34
f['chi_cw'] = 100 * (f['delta_core'] - 47.8) ** 2 / (8.314 * 310) + 0.34
f['Kam'] = np.exp(-(f['chi_dc'] + 0.5 * f['chi_cw']))
NUM = ['fratio', 'fphil', 'fphob', 'inter1', 'inter2', 'MolWt', 'LogP', 'TPSA', 'delta_drug', 'delta_core', 'chi_dc', 'chi_cw', 'Kam']
CAT = ['preparation_method', 'fam', 'data_source']
for c in CAT: f[c] = f[c].fillna('UNK').astype(str)
pre = ColumnTransformer([('n', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), NUM),
                         ('c', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))]), CAT)])
uidx = f.index[f['data_source'] == 'user_exp'].tolist()
loo = {}
for col, key in [('ee_percent', 'EE'), ('dl_percent', 'DL'), ('size_drug_nm', 'Size')]:
    d = f.dropna(subset=[col]); X = d[NUM + CAT]
    y = d[col].values.astype(float); lp = (key == 'Size')
    if lp: y = np.log1p(y)
    preds = {}
    for ui in uidx:
        if ui not in d.index: continue
        tr = d.index != ui
        pr = clone(pre); Xa = pr.fit_transform(X[tr]); Xb = pr.transform(X.loc[[ui]])
        m = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, reg_lambda=3.0, n_jobs=-1)
        m.fit(Xa, y[tr]); p = float(m.predict(Xb)[0])
        pv = float(np.expm1(p)) if lp else p
        true = float(np.expm1(y[d.index == ui][0])) if lp else float(y[d.index == ui][0])
        preds[str(d.loc[ui, 'sample_id'])] = {'pred': round(pv, 2), 'true': round(true, 2), 'err': round(abs(pv - true), 2)}
    yt = np.array([v['true'] for v in preds.values()]); yp = np.array([v['pred'] for v in preds.values()])
    loo[key] = {'n': len(preds), 'R2_loo': round(float(r2_score(yt, yp)), 3), 'MAE_loo': round(float(mean_absolute_error(yt, yp)), 3), 'rows': preds}
    print(key, 'R2_loo:', loo[key]['R2_loo'], 'MAE_loo:', loo[key]['MAE_loo'], flush=True)
    pr = clone(pre); Xa = pr.fit_transform(X)
    m = XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.05, reg_lambda=3.0, n_jobs=-1); m.fit(Xa, y)
    pickle.dump({'pre': pr, 'model': m, 'num': NUM, 'cat': CAT, 'log1p': lp}, open(f'models/v6_{key}.pkl', 'wb'))
json.dump(loo, open('reports/v6_loo.json', 'w'), indent=2)
print('SAVED models/v6_*.pkl reports/v6_loo.json')
