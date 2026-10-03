"""DeepMicelle desktop app: left sidebar inputs, right output panel."""
import json
import os
import sys
import tkinter as tk
from tkinter import messagebox

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import predict as P  # noqa: E402

CFG = json.load(open(os.path.join(BASE, 'config_constants.json'), encoding='utf-8'))
Q = json.load(open(os.path.join(BASE, 'reports/v6_metrics.json'), encoding='utf-8'))
DELTA = CFG['block_delta']
WATER = CFG['water_delta']

BG, CARD, TXT, ACC, INBG = '#0E1117', '#161B22', '#F0F6FC', '#58A6FF', '#21262D'

root = tk.Tk()
root.title('DeepMicelle v6')
root.geometry('1020x700')
root.configure(background=BG)
E = {}

side = tk.Frame(root, background=CARD, padx=14, pady=10, width=300)
side.pack(side='left', fill='y', padx=10, pady=10)
side.pack_propagate(False)
main = tk.Frame(root, background=BG, padx=14, pady=10)
main.pack(side='right', fill='both', expand=True, padx=(0, 10), pady=10)

tk.Label(side, text='输入', font=('Microsoft YaHei', 14, 'bold'),
         background=CARD, foreground=ACC).pack(anchor='w', pady=(0, 8))


def lab(text):
    tk.Label(side, text=text, background=CARD, foreground='#9AA4B2',
             font=('Microsoft YaHei', 9)).pack(anchor='w', pady=(6, 0))


def entry(key, width=26):
    v = tk.StringVar()
    tk.Entry(side, textvariable=v, width=width, background=INBG, foreground=TXT,
             insertbackground=TXT, relief='flat').pack(anchor='w', pady=2, ipady=4)
    E[key] = v
    return v


def combo(key, values):
    v = tk.StringVar(value=values[0])
    m = tk.OptionMenu(side, v, *values)
    m.config(width=22, background=INBG, foreground=TXT, highlightthickness=0, relief='flat')
    m.pack(anchor='w', pady=2)
    E[key] = v
    return v


lab('介质')
combo('medium', ['pure_water', 'PBS pH7.4', 'PBS pH5.5', 'glycerol_water'])
lab('介质常数 epsilon')
entry('eps')
lab('介质温度 (C)')
entry('tmed')
lab('嵌段家族')
combo('fam', ['PCL', 'PLA', 'PLGA'])
lab('亲水Mw (Da)')
entry('fphil')
lab('疏水Mw (Da)')
entry('fphob')
lab('疏/亲比')
entry('fratio')
lab('制备方法')
combo('prep', ['film_hydration', 'solvent_evaporation', 'dialysis', 'sonication'])
lab('水化温度 (C) / 体积 (mL) / 时间 (min)［仅薄膜法］')
entry('ht')
entry('hv')
entry('htime')
lab('药物名')
combo('drug', ['Doxorubicin', 'Paclitaxel', 'Methotrexate', '其它'])
lab('药物 Mw (Da)')
entry('Mw')
lab('LogP')
entry('LogP')
lab('TPSA')
entry('TPSA')

tk.Label(main, text='DeepMicelle 预测输出', font=('Microsoft YaHei', 16, 'bold'),
         background=BG, foreground=TXT).pack(anchor='w', pady=(0, 4))
tk.Label(main, text='chi_dc<2 视为相容可成胶；EE 仅趋势参考',
         background=BG, foreground='#9AA4B2').pack(anchor='w', pady=(0, 10))

cards = {}
for t, lab in [('EE', '包封率 EE %'), ('DL', '载药量 DL %'), ('Size', '粒径 Size nm'), ('logCMC', 'logCMC')]:
    f = tk.Frame(main, background=CARD, padx=16, pady=12)
    f.pack(fill='x', pady=5)
    tk.Label(f, text=lab, font=('Microsoft YaHei', 11), background=CARD, foreground='#9AA4B2').pack(anchor='w')
    val = tk.Label(f, text='—', font=('Consolas', 26, 'bold'), background=CARD, foreground=ACC)
    val.pack(anchor='w')
    det = tk.Label(f, text='', font=('Microsoft YaHei', 9), background=CARD, foreground='#9AA4B2',
                   justify='left')
    det.pack(anchor='w')
    cards[t] = (val, det)

chi_line = tk.Label(main, text='', font=('Consolas', 12, 'bold'), background=BG, foreground=TXT)
chi_line.pack(anchor='w', pady=(8, 0))
flag_line = tk.Label(main, text='', background=BG, foreground='#F85149', wraplength=600, justify='left')
flag_line.pack(anchor='w')


def need(key):
    try:
        return float(E[key].get())
    except Exception:
        return None


def run():
    vals = {k: need(k) for k in ['eps', 'tmed', 'fphil', 'fphob', 'fratio', 'Mw', 'LogP', 'TPSA']}
    prep = E['prep'].get()
    if prep == 'film_hydration':
        for k in ['ht', 'hv', 'htime']:
            vals[k] = need(k)
    miss = [k for k, v in vals.items() if v is None]
    if miss:
        messagebox.showerror('缺少输入', '请填写：' + '、'.join(miss))
        return
    dc = DELTA[E['fam'].get()]
    _lp = max(-2.0, min(6.0, vals['LogP']))
    _den = max(80.0, min(900.0, vals['Mw'] / 1.2))
    dd = float(np.sqrt((8000.0 + _lp * 1500.0 + vals['TPSA'] * 180.0) / _den))
    T = 273.15 + vals['tmed']
    cdc = 100 * (dd - dc) ** 2 / (8.314 * T) + 0.34
    ccw = 100 * (dc - WATER) ** 2 / (8.314 * T) + 0.34
    row = {'fratio': vals['fratio'], 'fphil': vals['fphil'], 'fphob': vals['fphob'],
           'inter1': vals['fphob'] / (vals['fphil'] + 1), 'inter2': vals['fratio'] * np.log1p(vals['fphob']),
           'MolWt': vals['Mw'], 'LogP': vals['LogP'], 'TPSA': vals['TPSA'],
           'delta_drug': dd, 'delta_core': dc, 'chi_dc': cdc, 'chi_cw': ccw,
           'Kam': float(np.exp(-(cdc + 0.5 * ccw))), 'preparation_method': prep,
           'fam': E['fam'].get(), 'data_source': 'user_exp'}
    chi_line.config(text=f"chi_dc={cdc:.2f}  chi_cw={ccw:.2f}  Kam={row['Kam']:.2e}  "
                         + ('可成胶' if cdc < 2 else '域外：可能不成胶'))
    flags_all = []
    for t in ['EE', 'DL', 'Size', 'logCMC']:
        r = P.predict(t, row)
        val, det = cards[t]
        val.config(text=str(r['pred']))
        det.config(text=f"90%区间 {r['interval_90']}   MAE {r['MAE_cv']}")
        flags_all += r['flags']
    seen = list(dict.fromkeys(flags_all))
    flag_line.config(text='\n'.join('提示: ' + f for f in seen))


tk.Button(side, text='预 测', font=('Microsoft YaHei', 12, 'bold'),
          background=ACC, foreground='#0E1117', relief='flat', padx=20, pady=6,
          command=run).pack(anchor='w', pady=12)
root.mainloop()
