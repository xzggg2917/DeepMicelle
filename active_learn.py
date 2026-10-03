import pandas as pd, numpy as np, pickle, itertools
# candidate grid around under-sampled region: small hydrophobic core + polar drugs + film_hydration
fams=['PCL','PLA','PLGA']; preps=['film_hydration','solvent_evaporation','dialysis']
drugs=[('Doxorubicin',543.5,1.27,206.0),('Methotrexate',454.4,0.9,210.0),('Fluorescein',332.3,2.5,76.0),('Baicalin',446.4,1.1,187.0),('Paclitaxel',853.9,3.5,221.0)]
fphil_vals=[2000.0,5000.0]; fphob_vals=[872.0,1500.0,2500.0,4000.0]; ratio_vals=[0.44,0.7,1.0,1.6]
cands=[]
for (dn,mw,lp,tpsa),fam,fp,fb,r,pr in itertools.product(drugs,fams,fphil_vals,fphob_vals,ratio_vals,preps):
    dd=float(np.sqrt((8000+max(-2,min(6,lp))*1500+tpsa*180)/max(80,min(900,mw/1.2))))
    dc={'PCL':19.2,'PLA':20.2,'PLGA':21.0}[fam]
    cdc=100*(dd-dc)**2/(8.314*298.15)+0.34
    cands.append({'drug':dn,'Mw':mw,'LogP':lp,'TPSA':tpsa,'fam':fam,'fphil':fp,'fphob':fb,'fratio':r,'prep':pr,'delta_drug':dd,'delta_core':dc,'chi_dc':cdc,'chi_cw':100*(dc-47.8)**2/(8.314*298.15)+0.34})
C=pd.DataFrame(cands)
# training cloud distance
tr=pd.read_excel('data/cleaned/formulation_train_v3final.xlsx')
for c in ['hydrophilic_mw','hydrophobic_mw','hydrophobic_hydrophilic_ratio']:
    tr[c]=pd.to_numeric(tr[c],errors='coerce')
med=tr[['hydrophilic_mw','hydrophobic_mw','hydrophobic_hydrophilic_ratio']].median()
std=tr[['hydrophilic_mw','hydrophobic_mw','hydrophobic_hydrophilic_ratio']].std()+1e-6
Tn=((tr[['hydrophilic_mw','hydrophobic_mw','hydrophobic_hydrophilic_ratio']].fillna(med)-med)/std).values
Cn=((C[['fphil','fphob','fratio']].values-med.values)/std.values)
from scipy.spatial import cKDTree
d,_=cKDTree(Tn).query(Cn,k=1)
C['novelty']=d
# uncertainty proxy: far from data + chi in informative band (1<chi<5, boundary) + polar drug bonus
C['score']=C['novelty']+0.5*((C['chi_dc']>1)&(C['chi_dc']<5)).astype(float)+0.3*(C['TPSA']>150).astype(float)
# greedy diverse pick of 15
picked=[]; rest=C.copy()
first=rest['score'].idxmax(); picked.append(first); rest=rest.drop(first)
for _ in range(14):
    sub=rest.copy()
    D=((sub[['fphil','fphob','fratio','chi_dc']].values[:,None,:]-C.loc[picked][['fphil','fphob','fratio','chi_dc']].values[None,:,:])**2).sum(-1).min(1)
    sub['div']=D/D.max()
    sub['total']=sub['score']+sub['div']
    i=sub['total'].idxmax(); picked.append(i); rest=rest.drop(i)
R=C.loc[picked].reset_index(drop=True)
R['hyd_T']=60.0; R['hyd_V']=10.0; R['hyd_t']='30min'; R['medium']='pure_water'; R['eps']=80.2; R['tmed']=37.0
R.index=R.index+1; R.index.name='exp_no'
R.to_excel('reports/recommended_experiments_15.xlsx')
print(R[['drug','fam','fphil','fphob','fratio','prep','chi_dc','novelty','score']].round(2).to_string())
print('saved reports/recommended_experiments_15.xlsx')
