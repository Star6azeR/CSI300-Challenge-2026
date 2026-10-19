"""Prespecified exploratory comparison, with auditable public-data examples.
2022 has already been seen in the previous report. It is not an untouched test.
No data outside supplied Data directory are read for features or labels.
"""
import argparse
from pathlib import Path
import json,time,hashlib,platform,importlib.metadata
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from audit_and_pilot import load,make_features,ic_table,ic_summary
from detailed_features import build_features,build_labels,KEY,LOOKBACK

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'research/results_detailed'
TREE=dict(n_estimators=80,learning_rate=.04,num_leaves=15,max_depth=5,min_child_samples=300,
          reg_lambda=10,colsample_bytree=.85,subsample=.8,subsample_freq=1,random_state=42,n_jobs=4,
          verbosity=-1,deterministic=True,force_col_wise=True)
CAT=dict(iterations=150,depth=5,learning_rate=.04,l2_leaf_reg=10,loss_function='RMSE',
         random_seed=42,thread_count=4,verbose=False,allow_writing_files=False,task_type='CPU',
         bootstrap_type='No',random_strength=0,nan_mode='Min')
VARIANTS={
 'M0_constant':{'model':'constant','features':'base43','target':'none'},
 'M1_ridge':{'model':'ridge','features':'base43','target':'rank'},
 'M2_barra':{'model':'lgbm','features':'barra10','target':'rank'},
 'M3_raw_return':{'model':'lgbm','features':'base43','target':'raw'},
 'M4_rank':{'model':'lgbm','features':'base43','target':'rank'},
 'M5_history':{'model':'lgbm','features':'extended63','target':'rank'},
 'M6_lambdarank':{'model':'ranker','features':'base43','target':'10bins'},
 'M7_catboost':{'model':'catboost','features':'base43','target':'rank'},
 'M8_blend':{'model':'blend','features':'base43','target':'0.5*M4_rank+0.5*M7_rank'},
}

def summarize_ic(frame):
    vals=frame.rankic.dropna(); std=float(vals.std())
    return {'days':len(vals),'mean_rankic':float(vals.mean()),'rankic_std':std,
            'rankic_ir':float(vals.mean()/std) if std>0 else None,'positive_ratio':float(vals.gt(0).mean())}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUT);args=parser.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    if (out/'experiment_summary.json').exists():raise RuntimeError('Choose a new output directory to preserve this experiment.')
    versions={n:importlib.metadata.version(n) for n in ['pandas','numpy','pyarrow','scipy','lightgbm','catboost','scikit-learn','skd-backtest']}
    prereg={'purpose':'Detailed exploratory report; 2022 already observed; no tuning or early stopping in this comparison.',
      'variants':VARIANTS,'tree_parameters':TREE,'catboost_parameters':CAT,'ridge_alpha':100,
      'ranker_parameters':{'label_gain':list(range(10)),'lambdarank_truncation_level':300,'objective':'lambdarank'},
      'years':[2019,2020,2021,2022],'lookback':LOOKBACK,'versions':versions,'python':platform.python_version(),
      'stage':'written before any new model result'}
    (out/'design.json').write_text(json.dumps(prereg,ensure_ascii=False,indent=2))
    started=time.time()
    m,f,b=[load(name) for name in ['MarketData','Factor33_winsor','Barra_factor']]
    x,allcols=build_features(f,b,m);basecols=[c for c in allcols if not c.startswith('hist_')];bcols=[c for c in basecols if c.startswith('barra_')]
    labels=build_labels(m,b)
    samples=x.merge(labels[KEY+['entry_date','exit_date','future_return','target']],on=KEY,validate='one_to_one')
    samples.to_parquet(out/'samples.parquet',index=False)
    (out/'feature_columns.json').write_text(json.dumps({'base':basecols,'barra':bcols,'extended':allcols},ensure_ascii=False,indent=2))
    # Non-outcome-selected worked examples, a liquid stock and a missing-value case.
    chosen=20220104;code='SH600000'
    raw=m.loc[m['代码'].eq(code)&m['日期'].isin([20211230,20211231,20220104,20220105,20220106])]
    raw.to_csv(out/'example_price_path.csv',index=False)
    detail=labels.loc[labels['日期'].eq(chosen)&labels['代码'].eq(code)]
    detail.to_csv(out/'example_label.csv',index=False)
    fac=f.merge(b[KEY],on=KEY,how='inner',validate='one_to_one')
    fac.loc[fac['日期'].eq(chosen)&fac['代码'].eq(code)].to_csv(out/'example_raw_factors.csv',index=False)
    x.loc[x['日期'].eq(chosen)&x['代码'].eq(code)].to_csv(out/'example_features.csv',index=False)
    stats=fac.drop(columns=['日期','代码','名称']).quantile([.01,.5,.99]).T
    stats.columns=['p01','median','p99'];stats['missing']=fac.drop(columns=['日期','代码','名称']).isna().sum()
    stats.to_csv(out/'factor_statistics.csv')
    fac.loc[fac.ps_ttm.isna()].head(5).to_csv(out/'example_missing.csv',index=False)
    # As-of feature parity on edge and ordinary dates, including an index rebalance vicinity.
    parity=[];dates=sorted(m['日期'].unique())
    for day in [20160104,20160108,20160613,20211213,20220104,20220613,20221230]:
        pos=dates.index(day); visible=dates[max(0,pos-60):pos+1]
        subset={k:v[v['日期'].isin(visible)] for k,v in [('f',f),('b',b),('m',m)]}
        online,cs=build_features(subset['f'],subset['b'],subset['m'],calendar=visible)
        left=x[x['日期'].eq(day)].sort_values('代码')[allcols].to_numpy()
        right=online[online['日期'].eq(day)].sort_values('代码')[allcols].to_numpy()
        assert np.allclose(left,right,equal_nan=True,rtol=1e-9,atol=1e-9)
        diff=np.abs(left-right);parity.append({'date':day,'rows':len(left),'max_abs_error':float(np.nanmax(diff))})
    (out/'feature_parity.json').write_text(json.dumps(parity,indent=2))
    models=out/'models';models.mkdir(exist_ok=True)
    rows=[];all_ic=[];all_predictions=[]
    for year in [2019,2020,2021,2022]:
        train=samples[samples.exit_date.le((year-1)*10000+1231)&samples.future_return.notna()].copy()
        test=samples[samples['日期'].between(year*10000+101,year*10000+1231)].copy()
        eligible=test.future_return.notna()&test.exit_date.le(year*10000+1231)
        weight=1/train.groupby('日期')['代码'].transform('size');weight/=weight.mean()
        pred={};year_preds=test[KEY+['future_return','entry_date','exit_date']].copy()
        for name,spec in VARIANTS.items():
            t0=time.time();cols={'base43':basecols,'barra10':bcols,'extended63':allcols}[spec['features']]
            kind=spec['model'];model=None
            if kind=='constant':scores=np.zeros(len(test))
            elif kind=='blend':
                z=pd.DataFrame({'day':test['日期'].to_numpy(),'a':pred['M4_rank'],'b':pred['M7_catboost']})
                scores=(.5*z.groupby('day').a.rank(pct=True)+.5*z.groupby('day').b.rank(pct=True)).to_numpy()
            else:
                target=train.future_return if spec['target']=='raw' else train.target
                if kind=='ridge':
                    model=Ridge(alpha=100);model.fit(train[cols].fillna(.5),target,sample_weight=weight)
                    scores=model.predict(test[cols].fillna(.5))
                    if year==2022:np.savez(models/(name+'.npz'),coef=model.coef_,intercept=model.intercept_)
                elif kind=='catboost':
                    model=CatBoostRegressor(**CAT);model.fit(train[cols],target,sample_weight=weight)
                    scores=model.predict(test[cols])
                    if year==2022:model.save_model(str(models/(name+'.cbm')))
                elif kind=='ranker':
                    target=np.minimum(np.floor(train.target.to_numpy()*10).astype(int),9)
                    model=lgb.LGBMRanker(**TREE,objective='lambdarank',label_gain=list(range(10)),lambdarank_truncation_level=300)
                    groups=train.groupby('日期',sort=False).size().to_numpy()
                    assert np.all(train['日期'].to_numpy()[:-1]<=train['日期'].to_numpy()[1:])
                    model.fit(train[cols],target,group=groups,sample_weight=weight)
                    scores=model.predict(test[cols])
                    if year==2022:model.booster_.save_model(str(models/(name+'.txt')))
                else:
                    model=lgb.LGBMRegressor(**TREE);model.fit(train[cols],target,sample_weight=weight)
                    scores=model.predict(test[cols])
                    if year==2022:
                        model.booster_.save_model(str(models/(name+'.txt')))
                        importance=pd.DataFrame({'feature':cols,'gain':model.booster_.feature_importance('gain'),'split':model.booster_.feature_importance('split')}).sort_values('gain',ascending=False)
                        importance.to_csv(out/(name+'_importance.csv'),index=False)
            assert len(scores)==len(test) and np.isfinite(scores).all()
            pred[name]=scores;year_preds[name]=scores
            ic=ic_table(test[eligible],scores[eligible.to_numpy()]);ic['model']=name;ic['year']=year
            all_ic.append(ic)
            summary={'model':name,'year':year,'features':len(cols),'train_rows':len(train),'eval_rows':int(eligible.sum()),
                     'fit_predict_seconds':time.time()-t0,**summarize_ic(ic)}
            rows.append(summary);print(json.dumps(summary),flush=True)
            pd.DataFrame(rows).to_csv(out/'annual_results.csv',index=False)
        all_predictions.append(year_preds)
        year_preds.to_parquet(out/f'predictions_{year}.parquet',index=False)
    daily=pd.concat(all_ic,ignore_index=True);daily.to_csv(out/'daily_ic.csv',index=False)
    summaries=[]
    for name in VARIANTS:
        subset=daily[daily.model.eq(name)]
        summary={'model':name,**summarize_ic(subset),'dev_mean_rankic':float(subset[subset.year.le(2021)].rankic.mean())}
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    predictions=pd.concat(all_predictions,ignore_index=True)
    # Per-day rank correlations between model scores, rather than pooled raw scales.
    ranked=predictions[list(VARIANTS)].groupby(predictions['日期']).rank(pct=True)
    ranked.corr().to_csv(out/'prediction_rank_correlation.csv')
    result={'annual':rows,'summary':summaries,'feature_count':len(allcols),'elapsed_seconds':time.time()-started,'design':prereg}
    (out/'experiment_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('FINISHED',time.time()-started,flush=True)

if __name__=='__main__':main()
