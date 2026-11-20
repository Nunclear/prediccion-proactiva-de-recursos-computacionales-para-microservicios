"""Publication plots from audited observations only; no synthetic result curves."""
import os,sys,json,hashlib
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from core import Series,mask_for,CASES
ANA=ROOT/'results/analysis';FIG=ROOT/'articulo_final/figuras'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':180})
COLORS={'DLinear':'#2866a0','E5':'#de7b22','A0':'#7d7d7d','A1':'#849b4b','A2':'#9068b1','A3':'#009887','PatchTST':'#af3d60','GRUD':'#5b4ea1'}
NAMES={'cpu_usage':'CPU','container_memory_usage_bytes':'Memory'}
PATTERN_LABEL={'disperso':'Scattered','bloques':'Short blocks','correlacionado':'Correlated','prolongado':'Prolonged blocks','MAR_lag':'Lag-dependent MAR'}
manifest=[]
def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(fig,key,title,source,description):
 fig.suptitle(title,fontsize=11,fontweight='bold',y=.995)
 fig.text(.01,.005,f'20261110 | {source}',fontsize=7.5,color='#555555')
 fig.tight_layout(rect=[0,.045,1,.95])
 for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{key}.{ext}',bbox_inches='tight')
 manifest.append(dict(id=key,title=title,source=source,description=description,pdf_sha256=filehash(FIG/f'{key}.pdf')));plt.close(fig)
def box(ax,x,y,text,width=.21,height=.23):
 p=FancyBboxPatch((x-width/2,y-height/2),width,height,boxstyle='round,pad=.01',facecolor='#eef4f6',edgecolor='#31576b',linewidth=1.1);ax.add_patch(p);ax.text(x,y,text,ha='center',va='center',fontsize=9)
def arrow(ax,x1,y1,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','color':'#31576b','lw':1.1})
def diagram(key,title,nodes,edges,description):
 fig,ax=plt.subplots(figsize=(8,3));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
 for x,y,s in nodes:box(ax,x,y,s)
 for edge in edges:arrow(ax,*edge)
 save(fig,key,title,'configs/protocolo_congelado.yaml; src/models.py',description)
def original_positive(df):return df[df.q.gt(0)&df.pattern.isin(['disperso','bloques'])]
def primary(df):return original_positive(df[(df.dataset=='new_services')&(df.L==12)&(df.h_steps==1)])
def draw_curves(ax,df,x,models):
 for model in models:
  g=df[df.model==model].groupby(x).MASE.mean().sort_index()
  ax.plot(g.index,g.values,'o-',label=model,color=COLORS.get(model),lw=1.5,ms=4)
 ax.set_ylabel('MASE (dimensionless)');ax.grid(alpha=.2)
def make_diagrams():
 diagram('01_methodology','Study flow and separation of evidence',[(.13,.70,'Historical 28\nsource services'),(.43,.70,'Source-only tuning\nand weight fitting'),(.75,.70,'Freeze protocol\nand models'),(.43,.25,'New 32 services\n70/10/20 time cuts'),(.75,.25,'Alibaba 24 h case\n6h calibration')],[(.25,.70,.31,.70),(.55,.70,.63,.70),(.75,.57,.75,.38),(.68,.57,.48,.38)],'Historical evidence motivates the design. Only source training/validation determines weights and tuning. Both target domains receive frozen weights with prefix calibration; external evidence is exploratory.')
 diagram('02_dlinear','DLinear with fixed temporal decomposition',[(.13,.5,'Causal filled\nwindow v'),(.43,.74,'Fixed MA25\ntrend Sv'),(.43,.25,'Residual\n(I-S)v'),(.76,.5,'Two affine heads\nthen sum')],[(.25,.55,.31,.72),(.25,.45,.31,.27),(.55,.74,.65,.55),(.55,.25,.65,.45)],'The historical DLinear implementation uses replicated padding and separate trend/residual heads. Their sum defines an affine function when S is fixed.')
 diagram('03_a3','A3: affine prediction with two availability descriptors',[(.13,.5,'Causal value\nmask and age'),(.44,.77,'Affine value head\nL+1 parameters'),(.44,.25,'Missing fraction\nand last log-age'),(.78,.5,'Sum with two\nscalar coefficients')],[(.25,.55,.32,.75),(.25,.45,.32,.27),(.56,.77,.67,.55),(.56,.25,.67,.45)],'A0 is the collapsed affine head; A1 adds missing fraction, A2 adds last age, A3 both. All start with the same value function as seeded DLinear. L12 yields15 A3 parameters per resource.')
def main():
 assert (ANA/'COMPLETED.json').exists(),'Audit and analysis must finish before figures'
 FIG.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(ANA/'metrics_all.csv.gz');p=primary(df);summ=p.groupby(['resource','model']).MASE.mean()
 make_diagrams()
 data=Series(ROOT/'data/processed/chronograph_holdout.npz')
 fig,axes=plt.subplots(2,1,figsize=(7.1,4.8),gridspec_kw={'height_ratios':[.8,2]})
 ax=axes[0]
 for lo,hi,color,label in [(0,5603,'#b7d2e8','Training prefix'),(5603,6404,'#edd8a8','Validation period'),(6404,8005,'#badbcb','Test period')]:ax.barh(0,hi-lo,left=lo,color=color,height=.45,label=label)
 ax.axvline(5907,color='#b04343',ls='--',lw=1);ax.set_yticks([]);ax.set_xlabel('Chronological row index');ax.legend(loc='lower left',bbox_to_anchor=(0,1),ncol=3,fontsize=8)
 ax=axes[1];ix=np.arange(6404,6433);ax.plot(ix,data.values[ix,0,0],color='#b4b4b4',label='Actual series (illustration)')
 hist=np.arange(6404,6416);ax.plot(hist,data.values[hist,0,0],'o-',color='#2866a0',ms=3,label='L12 input, available by t-1');ax.scatter([6416],data.values[[6416],0,0],color='#de7b22',label='Target at t',zorder=5);ax.axvline(6415.5,color='#777',ls=':');ax.set_xlabel('Chronological row index');ax.set_ylabel('CPU usage (publisher scale)');ax.legend(fontsize=8);ax.grid(alpha=.2)
 save(fig,'04_temporal_preparation','Chronological preparation and a direct endpoint','chronograph_holdout.npz; protocol L12 H1','The upper panel shows cut positions and the validation-period gap after row5907. The lower panel illustrates the first valid test history and target for service002. Future values shown in grey are excluded from inputs.')
 fig,axes=plt.subplots(5,1,figsize=(7.1,6.2),sharex=True)
 for ax,pattern in zip(axes,['disperso','bloques','correlacionado','prolongado','MAR_lag']):
  m=mask_for(data,'test',20,pattern,90001);image=(~m[6404:6660]).T
  ax.imshow(image,aspect='auto',interpolation='nearest',cmap='Greys',vmin=0,vmax=1);ax.set_ylabel('Service\nindex');ax.set_title(PATTERN_LABEL[pattern]+f' (realized full-test loss: {100*(~m[6404:]).mean():.2f}%)',fontsize=9,loc='left')
 axes[-1].set_xlabel('Steps from test boundary (30 min per step); black = unavailable')
 save(fig,'05_missingness','Paired synthetic telemetry interruptions','mask seed90001; src/core.py; first256 test steps','The examples use actual generated masks, not measured outages. Group correlation and duration are imposed mechanisms. MAR_lag probability varies; its nominal label does not assert exactly20% realized loss.')
 models=sorted(p.model.unique(),key=lambda m:summ.loc[('cpu_usage',m)])
 fig,axes=plt.subplots(1,2,figsize=(7.1,5.4),sharey=True)
 for ax,(resource,label) in zip(axes,NAMES.items()):
  vals=[summ.loc[(resource,m)] for m in models];ax.barh(np.arange(len(models)),vals,color=[COLORS.get(m,'#b2b7bd') for m in models]);ax.set_yticks(range(len(models)),models);ax.set_ylim(len(models)-.5,-.5);ax.set_title(label);ax.set_xlabel('MASE (dimensionless)');ax.grid(axis='x',alpha=.2)
 save(fig,'06_models','Mean error on new services','analysis/primary_summary.csv; L12 H1; six positive-loss cases','Equal weight across services, six original positive-loss conditions and mask realizations, then training seeds. Sorting by CPU is display-only; model selection used source validation.')
 fig,axes=plt.subplots(2,2,figsize=(7.1,5.8),sharex=True)
 base=df[(df.dataset=='new_services')&(df.L==12)&(df.h_steps==1)]
 chosen=['DLinear','E5','A0','A3','GRUD','PatchTST']
 for row,(resource,label) in enumerate(NAMES.items()):
  for col,pattern in enumerate(['disperso','bloques']):
   g=base[(base.resource==resource)&((base.pattern==pattern)|base.q.eq(0))];draw_curves(axes[row,col],g,'q',chosen);axes[row,col].set_title(label+' / '+PATTERN_LABEL[pattern]);axes[row,col].set_xlabel('Nominal missing telemetry (%)');axes[row,col].set_xticks([0,10,20,40])
 axes[0,0].legend(fontsize=7.5,ncol=2)
 save(fig,'07_loss','Error across original loss conditions','analysis/metrics_all.csv.gz; L12 H1; new_services','The complete condition is shared at q0. The figure displays six prespecified families; all families and conditions remain in machine-readable tables. Curves are descriptive means.')
 common=pd.read_csv(ANA/'common_origin_metrics.csv.gz')
 common=original_positive(common);four=['DLinear','E5','A3','PatchTST']
 for key,title,subset,x,xlabel in [('08_horizon','Direct forecast horizons',common[common.L==12],'h_minutes','Forecast horizon (min)'),('09_history','Historical window length',common[common.h_steps==1],'L','History length (30-min steps)')]:
  fig,axes=plt.subplots(1,2,figsize=(7.1,3.6))
  for ax,(resource,label) in zip(axes,NAMES.items()):draw_curves(ax,subset[subset.resource==resource],x,four);ax.set_title(label);ax.set_xlabel(xlabel)
  axes[0].legend(fontsize=8)
  save(fig,key,title,'analysis/common_origin_metrics.csv.gz; 3 seeds / 5 masks / common targets','All points use the same target-row intersection, three seeds and first five masks; the primary point here differs from the full primary endpoint. This is a descriptive star design; each endpoint uses a separately trained direct model, not recursive forecasts.')
 fig,axes=plt.subplots(1,2,figsize=(7.1,3.6));rng=np.random.default_rng(61110)
 for ax,(resource,label) in zip(axes,NAMES.items()):
  for j,m in enumerate(four):
   a=p[(p.resource==resource)&(p.model==m)].groupby('seed').MASE.mean();ax.scatter(j+rng.uniform(-.08,.08,len(a)),a,color=COLORS[m],s=18,alpha=.8)
   ax.plot([j-.18,j+.18],[a.mean()]*2,color='black',lw=1.4)
  ax.set_xticks(range(len(four)),four,rotation=25);ax.set_ylabel('MASE (dimensionless)');ax.set_title(label);ax.grid(axis='y',alpha=.2)
 save(fig,'10_seed_variability','Ten independently initialized fits per neural family','analysis/metrics_all.csv.gz; primary positive-loss endpoint','Each dot is one seeded training run averaged across the fixed services, six conditions and twenty masks; a black line marks the seed mean. Dots are not independent infrastructures.')
 fig,axes=plt.subplots(1,2,figsize=(7.1,3.6));abl=['A0','A1','A2','A3']
 for ax,(resource,label) in zip(axes,NAMES.items()):
  means=[];sds=[]
  for m in abl:
   a=p[(p.resource==resource)&(p.model==m)].groupby('seed').MASE.mean();means.append(a.mean());sds.append(a.std())
  ax.errorbar(range(4),means,yerr=sds,fmt='o',capsize=3,color='#2866a0');ax.set_xticks(range(4),['A0\nvalue','A1\n+fraction','A2\n+age','A3\n+both']);ax.set_ylabel('MASE (dimensionless)');ax.set_title(label);ax.grid(alpha=.2)
 save(fig,'11_ablations','Availability descriptor ablation','analysis/metrics_all.csv.gz; L12 H1; ten seeds','Error bars denote standard deviation across training-seed means, not confidence or prediction intervals. Each descriptor is independently evaluated while preserving common initial value functions.')
 external=original_positive(df[df.dataset=='alibaba_transfer'])
 if len(external):
  fig,axes=plt.subplots(1,2,figsize=(7.1,5.2),sharey=True)
  names=sorted(external.model.unique())
  for ax,(resource,label) in zip(axes,NAMES.items()):
   g=external[external.resource==resource].groupby('model').MASE.mean();vals=np.array([g.get(n,np.nan) for n in names]);ax.barh(range(len(names)),vals,color=[COLORS.get(n,'#b2b7bd') for n in names]);ax.set_yticks(range(len(names)),names);ax.set_ylim(len(names)-.5,-.5);ax.set_title(label);ax.set_xlabel('MASE (dimensionless)')
   if np.nanmax(vals)/max(np.nanmin(vals),1e-12)>30:ax.set_xscale('log');ax.set_xlabel('MASE (dimensionless, log scale)')
   ax.grid(axis='x',alpha=.2)
  save(fig,'12_external','Exploratory Alibaba calibrated transfer','analysis/metrics_all.csv.gz; Alibaba first24h; L12 H1','Published normalized pod utilization is averaged within completed intervals. The first6h calibrate scale and MASE; remaining18h are targets. Weights stay source-trained. GRU_graph is excluded and no infrastructure-population claim is made.')
 latency=pd.read_csv(ROOT/'experiments/benchmarks/latency.csv');params=latency.groupby('model').parameters.first()
 fig,axes=plt.subplots(1,2,figsize=(7.1,3.8))
 for ax,device in zip(axes,['cpu','cuda']):
  g=latency[(latency.device==device)&latency.batch.eq(1)]
  g=g[(g.method=='eager')&((g.model!='GRUD')|(device!='cuda'))|((g.model=='GRUD')&(g.method=='cuda_graph'))]
  for row in g.itertuples():
   ax.scatter(row.p50_ms,summ.loc[('cpu_usage',row.model)],s=28,color=COLORS.get(row.model,'#aeb7c2'),label=row.model)
   if row.model in ['DLinear','E5','A3','GRUD','PatchTST']:ax.annotate(row.model,(row.p50_ms,summ.loc[('cpu_usage',row.model)]),xytext=(3,4),textcoords='offset points',fontsize=7.5)
  ax.set_xscale('log');ax.set_xlabel('p50 model latency (ms, log scale)');ax.set_ylabel('CPU MASE (dimensionless)');ax.set_title(device.upper()+' / B1 / 32 services');ax.grid(alpha=.2)
 save(fig,'13_latency','Accuracy and measured inference cost','benchmarks/latency.csv; analysis/primary_summary.csv','Resident-tensor inference only with20 warmups,100 timed trials and CUDA synchronization. GRU-D CUDA uses verified replay, excluding graph capture. Shared-laptop observations do not measure an autoscaling pipeline.')
 fig,axes=plt.subplots(1,2,figsize=(7.1,3.6))
 for ax,(resource,label) in zip(axes,NAMES.items()):
  for model in models:
   ax.scatter(params[model],summ.loc[(resource,model)],s=28,color=COLORS.get(model,'#aeb7c2'))
   if model in ['DLinear','E5','A3','GRUD','PatchTST']:ax.annotate(model,(params[model],summ.loc[(resource,model)]),xytext={'A3':(-7,9),'DLinear':(2,-12),'E5':(3,5),'GRUD':(3,5),'PatchTST':(4,2)}[model],textcoords='offset points',fontsize=7.5)
  ax.set_xscale('symlog',linthresh=1);ax.set_xlabel('Trainable parameters (symlog; zero retained)');ax.set_ylabel('MASE (dimensionless)');ax.set_title(label);ax.grid(alpha=.2)
 save(fig,'14_parameters','Accuracy and parameter count','benchmarks/latency.csv; analysis/primary_summary.csv','Zero-parameter references are retained at zero using a symmetric-log axis. Fewer parameters alone do not imply a measured speedup. The axis is per resource, shared across services.')
 per=pd.read_csv(ANA/'paired_by_service.csv');per=per[per.reference=='DLinear']
 fig,axes=plt.subplots(1,2,figsize=(7.1,6.5),sharey=True);services=data.services
 for ax,(resource,label) in zip(axes,NAMES.items()):
  g=per[per.resource==resource].set_index('service').loc[services];ax.barh(range(len(services)),g.MASE_difference,color=np.where(g.MASE_difference.ge(0),'#009887','#af3d60'));ax.set_yticks(range(len(services)),services,fontsize=8);ax.set_ylim(len(services)-.5,-.5);ax.axvline(0,color='black',lw=.7);ax.set_xlabel('MASE: DLinear - A3');ax.set_title(label);ax.grid(axis='x',alpha=.2)
 save(fig,'15_services','Heterogeneity across all 32 new services','analysis/paired_by_service.csv; primary positive-loss endpoint','Positive bars favor A3 and negative bars favor DLinear. Services remain fixed, connected study cases; the figure does not treat them as independent population replicates.')
 sources={str(p.relative_to(ROOT)):filehash(p) for p in [ANA/'primary_summary.csv',ANA/'metrics_all.csv.gz',ANA/'paired_by_service.csv',ROOT/'experiments/benchmarks/latency.csv',ROOT/'configs/protocolo_congelado.yaml']}
 (FIG/'manifest.json').write_text(json.dumps(dict(figures=manifest,sources_sha256=sources),indent=2))
 print('FIGURES_CREATED',len(manifest),flush=True)
if __name__=='__main__':main()
