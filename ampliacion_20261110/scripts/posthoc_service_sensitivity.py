"""Exploratory, outcome-informed sensitivity. Not part of frozen primary inference."""
from pathlib import Path
import json,math
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
files=sorted((ROOT/'experiments/baseline/L12_H1/test/new_services').glob('metrics_r1_rep*.csv'))
assert len(files)==20
columns=['model','resource','seed','q','pattern','service','MASE']
df=pd.concat([pd.read_csv(p,usecols=columns) for p in files],ignore_index=True)
df=df[df.model.isin(['E5','A3'])&df.q.gt(0)&df.pattern.isin(['disperso','bloques'])]
columns.append('realization')
df=pd.concat([pd.read_csv(p,usecols=columns) for p in files],ignore_index=True)
df=df[df.model.isin(['E5','A3'])&df.q.gt(0)&df.pattern.isin(['disperso','bloques'])]
a=df[df.model=='E5'].merge(df[df.model=='A3'],on=['resource','seed','q','pattern','service','realization'],suffixes=('_E5','_A3'),validate='one_to_one')
a['difference']=a.MASE_E5-a.MASE_A3
by_service=a.groupby('service').difference.mean().sort_values()
assert len(by_service)==32
largest=str(by_service.index[-1]);top3=list(by_service.index[-3:]);out=[]
for label,excluded in [('all',[]),('without_largest',[largest]),('without_top_three',top3)]:
 g=a[~a.service.isin(excluded)].groupby('seed').difference.mean().sort_index();n=len(g);k=int((g>0).sum());p=min(1,2*sum(math.comb(n,j) for j in range(min(k,n-k)+1))/2**n)
 out.append(dict(scope=label,excluded_services=excluded,service_count=int(a.service.nunique()-len(excluded)),mean_difference=float(g.mean()),positive_seed_means=k,seed_count=n,exploratory_unadjusted_sign_p=float(p),seed_differences=g.tolist()))
result=dict(status='complete',timing='After primary target inspection, outcome-informed and exploratory only; not a substitute for the four frozen primary tests.',reference='E5',proposed='A3',resource='container_memory_usage_bytes',services_favoring_A3=int((by_service>0).sum()),total_services=len(by_service),largest_service=largest,largest_service_mean_difference=float(by_service.iloc[-1]),largest_share_of_aggregate_mean=float(by_service.iloc[-1]/(32*by_service.mean())),summaries=out)
(ROOT/'results/analysis/posthoc_service_sensitivity.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
