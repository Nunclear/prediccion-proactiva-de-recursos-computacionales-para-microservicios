"""Build references from verified Crossref records and inspected primary arXiv pages."""
from pathlib import Path
import json,urllib.request,time,html,re
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
def esc(s):
 return html.unescape(str(s)).replace('&','\\&').replace('%','\\%').replace('_','\\_')
entries=[];records=[]
def add(kind,key,**fields):
 entries.append('@'+kind+'{'+key+',\n'+',\n'.join('  '+k+' = {'+str(v)+'}' for k,v in fields.items() if v is not None)+'\n}\n')
old=json.loads((BASE/'revision_20261029/docs/CROSSREF_METADATOS.json').read_text())
recent=json.loads((ROOT/'docs/crossref_RSL.json').read_text())
cache={x['metadata']['DOI'].lower():x['metadata'] for x in old+recent if x.get('metadata') and (x.get('verified') or x.get('verified_title'))}
doi_keys={'10.1609/aaai.v37i9.26317':'dlinear','10.1038/s41598-018-24271-9':'grud','10.1145/3637528.3672055':'ginar','10.1016/j.ijforecast.2006.03.001':'mase','10.3115/v1/D14-1179':'gru','10.1145/3542929.3563477':'powerprediction','10.1016/j.cosrev.2025.100791':'autoscaling2025','10.1109/tmlcn.2025.3553100':'aero','10.1007/s00607-025-01432-z':'aimapping','10.1109/CLOUD62652.2024.00030':'temposcale','10.1214/aos/1176347265':'blockbootstrap','10.1109/ICCCBDA69573.2026.11676496':'ibm2026','10.1145/3637528.3671508':'stamp'}
extra=ROOT/'docs/crossref_extra.json'
if extra.exists():
 for x in json.loads(extra.read_text()):
  if x.get('metadata'):cache[x['metadata']['DOI'].lower()]=x['metadata']
new=[]
for doi,key in doi_keys.items():
 try:
  if doi.lower() not in cache:
   req=urllib.request.Request('https://api.crossref.org/works/'+doi,headers={'User-Agent':'DLinear-research-metadata/1.0'})
   with urllib.request.urlopen(req,timeout=25) as f:cache[doi.lower()]=json.load(f)['message']
   time.sleep(1.5)
  m=cache[doi.lower()];assert m['DOI'].lower()==doi.lower()
  new.append(dict(doi=doi,metadata=m));year=m['published']['date-parts'][0][0]
  authors=' and '.join(esc(x.get('family',''))+', '+esc(x.get('given','')) for x in m.get('author',[]))
  full_title=m['title'][0]+(': '+m['subtitle'][0] if m.get('subtitle') else '');title=esc(full_title);venue=esc(m.get('container-title',[''])[0]);kind='inproceedings' if m['type']=='proceedings-article' else 'article'
  fields=dict(author=authors,title='{'+title+'}',year=year,doi=doi,url='https://doi.org/'+doi,volume=m.get('volume'),number=m.get('issue'),pages=m.get('page'),note='Metadata verified from Crossref; accessed 2026-11-10')
  fields['booktitle' if kind=='inproceedings' else 'journal']=venue;add(kind,key,**fields)
  records.append(dict(key=key,status='verified',doi=doi,title=full_title,year=year,primary_url='https://doi.org/'+doi))
 except Exception as ex:records.append(dict(key=key,status='unavailable',doi=doi,error=repr(ex)))
extra.write_text(json.dumps(new,indent=2,ensure_ascii=False))
manual=[
 ('patchtst','A Time Series is Worth 64 Words: Long-term Forecasting with Transformers','Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant',2023,'2211.14730','ICLR 2023; arXiv version 2'),
 ('timesnet','TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis','Wu, Haixu and Hu, Tengge and Liu, Yong and Zhou, Hang and Wang, Jianmin and Long, Mingsheng',2023,'2210.02186','ICLR 2023; arXiv version 3'),
 ('itransformer','iTransformer: Inverted Transformers Are Effective for Time Series Forecasting','Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng',2024,'2310.06625','ICLR 2024; arXiv version 4'),
 ('timemixer','TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting','Wang, Shiyu and Wu, Haixu and Shi, Xiaoming and Hu, Tengge and Luo, Huakun and Ma, Lintao and Zhang, James Y. and Zhou, Jun',2024,'2405.14616','ICLR 2024; arXiv version 1'),
 ('tide','Long-term Forecasting with TiDE: Time-series Dense Encoder','Das, Abhimanyu and Kong, Weihao and Leach, Andrew and Mathur, Shaan and Sen, Rajat and Yu, Rose',2023,'2304.08424','arXiv preprint, revised version 5 in 2024; publication status not inferred'),
 ('chronograph','ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset','Lutu, Adrian Catalin and Pintilie, Ioana and Burceanu, Elena and Manolache, Andrei',2025,'2509.04449','NeurIPS 2025 BERT2S workshop, as reported by authors; arXiv version 3'),
 ('conformal','A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification','Angelopoulos, Anastasios N. and Bates, Stephen',2021,'2107.07511','arXiv tutorial; revised version 6 in 2022'),
 ('radar2026','Dynamic Sampling for Telemetry in Microservices: A Reinforcement Learning and Entropy-Based Approach',"Alves, Renan Martins and Nobre, J{\\'e}ferson Campos and Wickboldt, Juliano Araujo",2026,'2609.31292','arXiv preprint; under review according to author record, not accepted publication'),
 ('ibn','IBN: An Interpretable Bidirectional-Modeling Network for Multivariate Time Series Forecasting with Variable Missing','Ma, Shusen and Zhang, Tianhao and Xia, Qijiu and Zhao, Yun-Bo',2025,'2509.07725','arXiv preprint'),
 ('taxonomy2026','Adaptive Management of Microservices in Dynamic Computing Environments: A Taxonomy and Future Directions','Chen, Ming and Islam, Muhammed Tawfiqul and Read, Maria Rodriguez and Buyya, Rajkumar',2026,'2604.25222','arXiv preprint'),
]
for key,title,authors,year,eprint,note in manual:
 add('misc',key,title='{'+title+'}',author=authors,year=year,eprint=eprint,archivePrefix='arXiv',url='https://arxiv.org/abs/'+eprint,note=note+'; primary page inspected 2026-11-10')
 records.append(dict(key=key,title=title,authors=authors,year=year,status='verified_primary_page',primary_url='https://arxiv.org/abs/'+eprint,publication_note=note))
add('inproceedings','bitgraph',author='Chen, Xiaodan and Li, Xiucheng and Liu, Bo and Li, Zhijun',title='{Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values}',booktitle='International Conference on Learning Representations',year=2024,url='https://proceedings.iclr.cc/paper_files/paper/2024/hash/7f3b25e39e0f0330435d1ffc1d6fa193-Abstract-Conference.html',note='Primary proceedings page verified 2026-11-10')
records.append(dict(key='bitgraph',title='Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values',authors='Xiaodan Chen; Xiucheng Li; Bo Liu; Zhijun Li',year=2024,status='verified_primary_proceedings',primary_url='https://proceedings.iclr.cc/paper_files/paper/2024/hash/7f3b25e39e0f0330435d1ffc1d6fa193-Abstract-Conference.html'))
add('article','holm',author='Holm, Sture',title='{A Simple Sequentially Rejective Multiple Test Procedure}',journal='Scandinavian Journal of Statistics',year=1979,volume=6,number=2,pages='65--70',url='https://www.jstor.org/stable/4615733',note='Primary issue contents verified 2026-11-10; no DOI inferred')
records.append(dict(key='holm',title='A Simple Sequentially Rejective Multiple Test Procedure',authors='Sture Holm',year=1979,status='verified_primary_issue_contents',primary_url='https://www.jstor.org/stable/i412579',article_url='https://www.jstor.org/stable/4615733'))
add('misc','alibaba2022',author='{Alibaba Clusterdata}',title='{Microservices Traces, Version 2022}',year=2022,url='https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022',note='Official dataset documentation; accessed 2026-11-10; download hashes in accompanying manifests')
(ROOT/'docs/BIBLIOGRAFIA_VERIFICADA.bib').write_text('\n'.join(entries))
(ROOT/'docs/bibliography_manifest.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
print(json.dumps([{k:x[k] for k in x if k in ['key','status','doi','title']} for x in records],indent=2,ensure_ascii=False))
