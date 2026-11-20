"""Record final PDF structural checks plus the completed human-visible render review."""
from pathlib import Path
import hashlib,json
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'articulo_final';TMP=ROOT/'tmp/pdfs'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected={'main':31,'revision_espanol':8};records={}
for stem,n in expected.items():
 p=P/f'{stem}.pdf';reader=PdfReader(p);assert len(reader.pages)==n and not reader.is_encrypted
 meta=reader.metadata;assert 'Carlos Augusto Mullisaca Condori' in str(meta.author) and 'Jhon Alex Gutiérrez Morales' in str(meta.author)
 full='\n'.join(x.extract_text() or '' for x in reader.pages)
 for token in ['Carlos Augusto Mullisaca Condori','Jhon Alex Gutiérrez Morales','carlos.mullisaca@upeu.edu.pe','alex.gutierrez@upeu.edu.pe']:
  assert token in full,(stem,token)
 for token in (['This research received no external funding.','The authors declare no conflicts of interest.'] if stem=='main' else ['Resumen','Financiación y conflictos de interés']):assert token in full,(stem,token)
 log=(P/f'{stem}.log').read_text()
 for bad in ['LaTeX Error','Undefined control sequence','Citation `','Reference `','Overfull \\hbox','Emergency stop']:
  assert bad not in log,(stem,bad)
 renders=sorted((TMP/('final_en' if stem=='main' else 'final_es')).glob('page-*.jpg'))
 assert len(renders)==n,(stem,len(renders),n)
 records[stem]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),pages=n,rendered_pages=len(renders),metadata_author=str(meta.author),metadata_title=str(meta.title),compile_log_no_errors_overfull_or_unresolved_citations=True)
figs=json.loads((P/'figuras/manifest.json').read_text())['figures'];assert len(figs)==15
for f in figs:
 for ext in ['pdf','svg','png']:assert (P/'figuras'/f"{f['id']}.{ext}").is_file()
 assert sha(P/'figuras'/f"{f['id']}.pdf")==f['pdf_sha256']
 assert len(PdfReader(P/'figuras'/f"{f['id']}.pdf").pages)==1
report=dict(status='passed',visual_review_completed=True,visual_review_method='All 31 English and 8 Spanish page renders inspected in contact sheets; full-size spot review of covers, numerical tables, references, figures and changed matrix page; all 15 figure PNGs inspected individually. No clipping, unreadable labels or content overflow observed.',pdfs=records,figure_pdfs_verified=15,limitations='Visual review and local compilation do not substitute for both authors final approval or a journal production proof.')
(ROOT/'results/pdf_qa.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps({k:v for k,v in report.items() if k!='pdfs'},indent=2))
