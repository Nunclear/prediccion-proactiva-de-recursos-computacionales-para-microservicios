"""Prueba técnica pequeña de entrenamiento/validación; no experimento confirmatorio."""
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
SERVICES=['service030','service033','service035','service040']
RESOURCES=['cpu_usage','container_memory_usage_bytes']
HISTORY=12
SEEDS=[101,202]
CASES=[(0,'disperso')]+[(q,p) for q in [10,20,40] for p in ['disperso','bloques']]
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)

def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

class ControlGRU(nn.Module):
    def __init__(self,outgoing,incoming):
        super().__init__()
        self.register_buffer('outgoing',torch.tensor(outgoing,dtype=torch.float32))
        self.register_buffer('incoming',torch.tensor(incoming,dtype=torch.float32))
        self.gru=nn.GRU(9,12,batch_first=True)
        self.head=nn.Linear(12,1)

    def forward(self,z):
        # z: lote, historia, servicio, [valor, disponibilidad, antigüedad]
        a=torch.einsum('ij,bhjd->bhid',self.outgoing,z)
        b=torch.einsum('ij,bhjd->bhid',self.incoming,z)
        x=torch.cat([z,a,b],dim=-1)
        batch,steps,nodes,_=x.shape
        x=x.permute(0,2,1,3).reshape(batch*nodes,steps,9)
        _,state=self.gru(x)
        return self.head(state[-1]).reshape(batch,nodes)

def normalize(a):
    degree=a.sum(1,keepdims=True)
    return np.divide(a,degree,out=np.zeros_like(a),where=degree!=0)

def segments(times,lo,hi):
    starts=[lo]+[i for i in range(lo+1,hi) if times[i]-times[i-1]!=1800]+[hi]
    return list(zip(starts,starts[1:]))

def mask_for(times,lo,hi,q,pattern,seed):
    observed=np.ones((len(times),len(SERVICES)),dtype=bool)
    counts=[]
    for n in range(len(SERVICES)):
        for segment,(a,b) in enumerate(segments(times,lo,hi)):
            rng=np.random.default_rng(seed+100*n+10000*segment)
            length=b-a; k=int(np.rint(q*length/100))
            priority=rng.random(length)
            if pattern=='disperso': chosen=np.argsort(priority,kind='stable')[:k]
            else:
                # Seis posiciones candidatas y una observada de separación.
                # La fase y el orden son comunes a todos los porcentajes.
                shift=int(rng.integers(0,7))
                blocks=[np.arange(max(0,c),min(length,c+6)) for c in range(-shift,length,7) if min(length,c+6)>max(0,c)]
                blocks.sort(key=lambda ix:float(priority[ix].min()))
                chosen=np.concatenate(blocks)[:k] if blocks else np.array([],dtype=int)
            observed[a+chosen,n]=False
            assert np.count_nonzero(~observed[a:b,n])==k
            runs=[]; run=0
            for yes in observed[a:b,n]:
                if not yes: run+=1
                elif run: runs.append(run); run=0
            if run:runs.append(run)
            if pattern=='bloques': assert max(runs,default=0)<=6
            counts.append({'servicio':SERVICES[n],'segmento':segment,'n':length,'ocultados':k,'racha_maxima':max(runs,default=0)})
    return observed,counts

def prepare(values,observed,times,lo,hi,mean,scale,median):
    z=np.zeros((len(times),len(SERVICES),3),dtype=np.float32)
    for a,b in segments(times,lo,hi):
        last=median.copy(); age=np.zeros(len(SERVICES))
        for t in range(a,b):
            yes=observed[t]
            last=np.where(yes,values[t],last)
            age=np.where(yes,0,age+1)
            z[t,:,0]=(last-mean)/scale
            z[t,:,1]=yes
            z[t,:,2]=np.log1p(age)/np.log1p(HISTORY)
    return z

def eligible(times,lo,hi):
    return np.array([t for t in range(lo+HISTORY,hi) if np.all(np.diff(times[t-HISTORY:t+1])==1800)],dtype=int)

def sample_indices(ix,count):
    return ix[np.linspace(0,len(ix)-1,min(count,len(ix)),dtype=int)]

def state_hash(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().numpy().tobytes())
    return h.hexdigest()

def run():
    source=OUT/'datos/muestra_conectada.json'
    raw=json.loads(source.read_text(encoding='utf-8'))
    times=np.array(raw[SERVICES[0]][RESOURCES[0]]['steps'],dtype=float)
    values=np.stack([np.stack([raw[s][r]['values'] for r in RESOURCES],axis=-1) for s in SERVICES],axis=1)
    assert values.shape==(8005,4,2) and np.isfinite(values).all()
    for s in SERVICES:
        for r in RESOURCES: assert np.array_equal(times,raw[s][r]['steps'])
    original_hash=digest(values)
    edges=list(csv.DictReader((ROOT/'Verificacion_viabilidad_y_brecha_20261024/auditoria_previa_20261018/ChronoGraph_edges_verificados.csv').open(encoding='utf-8-sig')))
    adjacency=np.zeros((4,4),dtype=np.float32)
    real_edges=[]
    for e in edges:
        if e['source'] in SERVICES and e['target'] in SERVICES:
            adjacency[SERVICES.index(e['source']),SERVICES.index(e['target'])]=1
            real_edges.append(e)
    assert len(real_edges)==3
    operators={'propia':(np.eye(4,dtype=np.float32),np.eye(4,dtype=np.float32)),'llamadas':(normalize(adjacency),normalize(adjacency.T))}
    cuts=[0,int(len(times)*.7),int(len(times)*.85),len(times)]
    train_idx=sample_indices(eligible(times,cuts[0],cuts[1]),32)
    val_idx=sample_indices(eligible(times,cuts[1],cuts[2]),16)
    assert train_idx.max()<cuts[1] and val_idx.min()-HISTORY>=cuts[1] and val_idx.max()<cuts[2]
    means=values[:cuts[1]].mean(axis=0)
    scales=np.maximum(values[:cuts[1]].std(axis=0),1e-8)
    medians=np.median(values[:cuts[1]],axis=0)
    paired=np.diff(times[:cuts[1]])==1800
    mase=np.abs(np.diff(values[:cuts[1]],axis=0))[paired].mean(axis=0)
    assert (mase>0).all()
    prepared={}; audits=[]; masks={}
    for partition,lo,hi,indices,seed in [('train',cuts[0],cuts[1],train_idx,70001),('validation',cuts[1],cuts[2],val_idx,80001)]:
        for q,p in CASES:
            mask,record=mask_for(times,lo,hi,q,p,seed)
            masks[partition,q,p]=mask
            audits.append({'particion':partition,'q':q,'patron':p,'sha256':digest(mask),'conteos':record})
        for r in range(2):
            xs=[];ys=[];groups=[]
            for c,(q,p) in enumerate(CASES):
                z=prepare(values[:,:,r],masks[partition,q,p],times,lo,hi,means[:,r],scales[:,r],medians[:,r])
                for t in indices:
                    xs.append(z[t-HISTORY:t]); ys.append((values[t,:,r]-means[:,r])/scales[:,r]); groups.append(c)
            prepared[partition,r]=(torch.tensor(np.stack(xs)),torch.tensor(np.stack(ys),dtype=torch.float32),np.array(groups))
    assert digest(values)==original_hash
    log=[]; learned={}; rows=[]

    def fit(kind,r,seed,replay=False):
        torch.manual_seed(seed)
        model=ControlGRU(*operators[kind])
        optim=torch.optim.Adam(model.parameters(),lr=.003)
        x,y,_=prepared['train',r]
        g=torch.Generator().manual_seed(seed+500)
        epoch_losses=[]
        for epoch in range(5):
            model.train(); order=torch.randperm(len(x),generator=g); losses=[]
            for batch in order.split(32):
                optim.zero_grad(set_to_none=True)
                prediction=model(x[batch]); loss=(prediction-y[batch]).square().mean()
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optim.step(); losses.append(float(loss.detach()))
            epoch_losses.append(float(np.mean(losses)))
        model.eval(); vx,vy,groups=prepared['validation',r]
        with torch.no_grad(): pred=model(vx).numpy()
        assert np.isfinite(pred).all()
        record={'variante':kind,'recurso':RESOURCES[r],'semilla':seed,'repeticion_identica':replay,'parametros':sum(p.numel() for p in model.parameters()),'perdida_entrenamiento_por_epoca':epoch_losses,'sha256_estado':state_hash(model),'sha256_prediccion_validacion':digest(pred)}
        log.append(record)
        if not replay:
            learned[kind,r,seed]=model
            for c,(q,pattern) in enumerate(CASES):
                delta=(pred[groups==c]-vy.numpy()[groups==c])*scales[:,r]
                rows.append({'variante':kind,'recurso':RESOURCES[r],'semilla':seed,'q':q,'patron':pattern,'MAE_validacion_tecnica':float(np.abs(delta).mean()),'RMSE_validacion_tecnica':float(np.sqrt((delta**2).mean())),'MASE_validacion_tecnica':float((np.abs(delta)/mase[:,r]).mean())})
        return record

    for r in range(2):
        for seed in SEEDS:
            for kind in operators:
                record=fit(kind,r,seed)
                print('Entrenado',kind,RESOURCES[r],seed,'parámetros',record['parametros'],flush=True)
    repeated=[]
    for r in range(2):
        for kind in operators:
            again=fit(kind,r,101,True)
            previous=next(z for z in log if z['variante']==kind and z['recurso']==RESOURCES[r] and z['semilla']==101 and not z['repeticion_identica'])
            same=again['sha256_estado']==previous['sha256_estado'] and again['sha256_prediccion_validacion']==previous['sha256_prediccion_validacion']
            assert same
            repeated.append({'variante':kind,'recurso':RESOURCES[r],'identica':same})
    controls=[]
    for r in range(2):
        x=prepared['validation',r][0][:1].clone()
        changed=x.clone();changed[:,:,0,0]+=3 # service030 es vecino de service033.
        unrelated=x.clone();unrelated[:,:,3,0]+=3 # service040 no es vecino directo de service030.
        for kind in operators:
            model=learned[kind,r,101]
            with torch.no_grad(): base=model(x); near=model(changed); far=model(unrelated)
            diff=float(torch.abs(base[0,1]-near[0,1])); invariant=float(torch.abs(base[0,0]-far[0,0]))
            assert invariant==0
            if kind=='propia': assert diff==0
            else: assert diff>1e-9
            controls.append({'variante':kind,'recurso':RESOURCES[r],'cambio_en_033_al_perturbar_030':diff,'cambio_en_030_al_perturbar_no_vecino_040':invariant})
    # Cambios posteriores al origen no deben alterar su historia preparada.
    origin=int(val_idx[0]-1); q=20; pattern='bloques'
    causality=[]
    for r in range(2):
        base=prepare(values[:,:,r],masks['validation',q,pattern],times,cuts[1],cuts[2],means[:,r],scales[:,r],medians[:,r])
        altered=values[:,:,r].copy(); altered[origin+1:]+=1e6
        after=prepare(altered,masks['validation',q,pattern],times,cuts[1],cuts[2],means[:,r],scales[:,r],medians[:,r])
        assert np.array_equal(base[origin-HISTORY+1:origin+1],after[origin-HISTORY+1:origin+1])
        # Perturbaciones de software, no datos sintéticos usados para entrenar.
        hidden=~masks['validation',q,pattern]
        altered=values[:,:,r].copy();altered[hidden]=1e8
        after_hidden=prepare(altered,masks['validation',q,pattern],times,cuts[1],cuts[2],means[:,r],scales[:,r],medians[:,r])
        assert np.array_equal(base,after_hidden)
        causality.append({'recurso':RESOURCES[r],'historia_invariante_a_futuro':True,'historia_invariante_a_valores_ocultados':True})
    assert digest(values)==original_hash
    (OUT/'resultados').mkdir(exist_ok=True)
    np.savez_compressed(OUT/'datos/muestra_y_operadores.npz',valores_originales=values,timestamps=times,adyacencia=adjacency,media_entrenamiento=means,escala_entrenamiento=scales,mediana_entrenamiento=medians,denominadores_MASE=mase,indices_train_tecnicos=train_idx,indices_val_tecnicos=val_idx)
    (OUT/'resultados/auditoria_mascaras.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'resultados/metricas_validacion_tecnica.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'estado':'prueba_tecnica_correcta','alcance':'Entrenamiento/validación pequeños; no selección de ganador ni contraste científico','entorno':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'dispositivo':'cpu','hilos':2,'algoritmos_deterministas':True},'servicios':SERVICES,'aristas_reales':real_edges,'restriccion_grafo':'Subgrafo inducido de cuatro servicios; vecinos externos excluidos, no población final','cortes':cuts,'historia':HISTORY,'horizonte_segundos':1800,'ventanas_train_por_condicion':len(train_idx),'ventanas_validacion_por_condicion':len(val_idx),'condiciones':CASES,'epocas':5,'pasos_optimizador_por_ajuste':5*((len(train_idx)*len(CASES)+31)//32),'ajustes_iniciales':8,'repeticiones_exactas':repeated,'controles_informacion':controls,'futuro_no_modifica_historia':True,'ocultados_no_modifican_historia_imputada':True,'valores_originales_intactos':True,'sha256_valores_originales':original_hash,'ventanas_prueba_formadas':0,'metricas_prueba_calculadas':0,'entrenamientos':log}
    result['controles_causalidad_por_recurso']=causality
    result['version_generador_bloques']='1.1: máximo seis mediciones consecutivas; separador observado; bloques de borde y último parcial; máscaras anidadas por porcentaje'
    result['codigo_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'resultados/resumen_prueba_B.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PRUEBA TÉCNICA TERMINADA; prueba final no utilizada',flush=True)

if __name__=='__main__':run()
