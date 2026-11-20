"""Independent full training; never imports weights from the source ZIP."""
import os,sys,json,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'src/baseline/reference'
sys.path.insert(0,str(HERE));os.environ.setdefault('CHRONOGRAPH_OUTPUT',str(ROOT/'results/reproduccion'))
import run
OUT=run.OUT;OUT.mkdir(parents=True,exist_ok=True)
(ROOT/'results/current_stage.json').write_text(json.dumps({'stage':'reproduccion_desde_cero','pid':os.getpid(),'progress':'results/reproduccion/progress.jsonl'}))
try:
 if not (OUT/'data_verified.npz').exists():subprocess.run([sys.executable,str(HERE/'audit.py')],check=True)
 before=list((OUT/'checkpoints').glob('*.pt'))
 if not (OUT/'fresh_start.json').exists():
  assert not before,'Nonempty checkpoint folder before initial run'
  (OUT/'fresh_start.json').write_text(json.dumps({'time':time.time(),'source_checkpoints_loaded':False,'source_archive':'archived, never used for initialization','full_training_required':True}))
 run.config();data=run.Data();run.checks(data);run.tune(data);run.fit_all(data);run.test_all(data)
 (OUT/'COMPLETED.json').write_text(json.dumps({'status':'complete','trained_from_scratch':True,'fresh_initialization_seeds':run.SEEDS,'models':run.MODELS}))
except BaseException as e:
 (OUT/'FAILURE.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e),'time':time.time(),'traceback':traceback.format_exc()}));raise
