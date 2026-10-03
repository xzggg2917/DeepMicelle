import streamlit as st, pickle, json, pandas as pd, numpy as np
st.set_page_config(page_title='DeepMicelle 预测系统', layout='wide')

st.markdown('<style>div[data-testid="stMetricValue"]{font-size:44px;font-weight:700}div[data-testid="stMetric"]{background:#111827;border-radius:12px;padding:16px}</style>', unsafe_allow_html=True)
st.title('DeepMicelle 胶束处方预测系统 v4')
Q=json.load(open('reports/v6_metrics.json',encoding='utf-8'))
# 注: 预测统一走 predict.py(v6 EE/DL/Size + v4 logCMC), 校准默认关闭
def num(label,unit=''):
    v=st.text_input(label+(' (%s)'%unit if unit else ''),value='',placeholder='请输入数字')
    try: return float(v)
    except Exception: return None
with st.sidebar:
    st.header('1. 介质条件')
    medium=st.selectbox('介质',['pure_water','PBS pH7.4','PBS pH5.5','glycerol_water'])
    eps=num('介质常数 epsilon')
    tmed=num('介质温度 C','C')
    st.header('2. 处方')
    fam=st.selectbox('疏水嵌段家族',['PCL','PLA','PLGA'])
    fphil=num('亲水Mw','Da'); fphob=num('疏水Mw','Da')
    fratio=num('亲疏水嵌段比(疏/亲)')
    prep=st.selectbox('制备方法',['film_hydration','solvent_evaporation','dialysis','sonication'])
    st.header('3. 水化参数')
    if prep=='film_hydration':
        ht=num('水化温度','C'); hv=num('水化体积','mL'); htime=num('水化时间','min')
    else:
        st.caption('非薄膜法无需水化参数'); ht,hv,htime=None,None,None
    st.header('4. 药物')
    drug=st.selectbox('药物名',['Paclitaxel','Doxorubicin','Docetaxel','Curcumin','Fluorescein','Azithromycin','Baicalin','其它(手动填)'])
    LIT={'Doxorubicin':(543.52,1.27,206.0),'Paclitaxel':(853.91,3.0,221.0),'Methotrexate':(454.44,1.85,211.0)}
    d0=LIT.get(drug,(None,None,None)); Mw=num('药物Mw','Da'); LogP=num('LogP'); TPSA=num('TPSA','A^2')
    st.caption('演示值已预填上组好数据，直接点预测即可。')
    go=st.button('预测',type='primary')
def chi(a,b,T): return 100*(a-b)**2/(8.314*(273.15+T))+0.34
if go:
    _miss=[k for k,v in [('eps',eps),('tmed',tmed),('fphil',fphil),('fphob',fphob),('fratio',fratio),('Mw',Mw),('LogP',LogP),('TPSA',TPSA)]+([('ht',ht),('hv',hv),('htime',htime)] if prep=='film_hydration' else []) if v is None]
    if _miss:
        st.error('请填写：'+'、'.join(_miss))
    else:
        delta_core={'PCL':19.2,'PLA':20.2,'PLGA':21.0}[fam]
        delta_drug=float(np.sqrt((8000+max(-2,min(6,LogP))*1500+TPSA*180)/max(80,min(900,Mw/1.2))))
        cdc=chi(delta_drug,delta_core,tmed); ccw=chi(delta_core,47.8,tmed); kam=float(np.exp(-(cdc+0.5*ccw)))
        row={'fratio':fratio,'fphil':fphil,'fphob':fphob,'inter1':fphob/(fphil+1),'inter2':fratio*np.log1p(fphob),'MolWt':Mw,'LogP':LogP,'TPSA':TPSA,'delta_drug':delta_drug,'delta_core':delta_core,'chi_dc':cdc,'chi_cw':ccw,'Kam':kam,'preparation_method':prep,'fam':fam,'data_source':'user_exp'}
        st.subheader('chi_dc=%.2f chi_cw=%.2f Kam=%.2e (数量级比较用)' % (cdc,ccw,kam))
        if cdc>2: st.error('域外：chi_dc>2，可能不成胶，请实验验证')
        else: st.success('可成胶：chi_dc<2')
        if prep=='film_hydration': st.write('水化条件：%.1f C / %.1f mL / %.1f min' % (ht,hv,htime))
        st.write('药物：%s 介质：%s epsilon=%.1f T=%.1fC' % (drug,medium,eps,tmed))
        cols=st.columns(4)
        for i,(t,lab) in enumerate([('EE','包封率%'),('DL','载药量%'),('Size','粒径nm'),('logCMC','logCMC')]):
            import predict as P
            r=P.predict(t,row)
            mae=r['MAE_cv']; r2=Q[t]['R2_stratified']
            with cols[i]:
                st.metric(lab,r['pred'],'pm %.2f' % Q[t]['q90'])
                st.caption('90pct区间 [%.1f, %.1f] MAE %.1f R2 %.2f' % (r['interval_90'][0],r['interval_90'][1],mae,r2))
                if r['calibration_weight']>0:
                    st.caption('未校准原值 %.2f (锚点w=%.2f)' % (r['pred_raw'],r['calibration_weight']))
                for fl in r['flags']:
                    st.caption('提示: '+fl)
