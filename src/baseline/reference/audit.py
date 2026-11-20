"""Verify the supplied cohort against the complete, locally cloned LFS object."""
import csv
import os
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source/Viabilidad_integral_AB_20261024'
RESOURCES = ['cpu_usage', 'container_memory_usage_bytes']

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def stream_members(path):
    decoder = json.JSONDecoder()
    with open(path, encoding='utf-8') as f:
        buf = ''; pos = 0; started = False
        while True:
            if len(buf) - pos < 1024**2:
                buf = buf[pos:] + f.read(4 * 1024**2); pos = 0
            while pos < len(buf) and buf[pos].isspace(): pos += 1
            if not started:
                assert buf[pos] == '{'; pos += 1; started = True
            while pos < len(buf) and (buf[pos].isspace() or buf[pos] == ','): pos += 1
            if buf[pos] == '}': return
            start = pos
            try:
                key, pos = decoder.raw_decode(buf, pos)
                while buf[pos].isspace(): pos += 1
                assert buf[pos] == ':'; pos += 1
                while buf[pos].isspace(): pos += 1
                value, pos = decoder.raw_decode(buf, pos)
            except (json.JSONDecodeError, IndexError):
                extra = f.read(4 * 1024**2)
                if not extra: raise
                buf = buf[start:] + extra; pos = 0
                continue
            yield key, value

def main():
    out = Path(os.environ.get('CHRONOGRAPH_OUTPUT','/workspace/results/reproduccion')); out.mkdir(exist_ok=True)
    selection = json.loads((SOURCE / 'cohorte_preseleccionada.json').read_text())
    cohort = json.loads((SOURCE / 'datos/cohorte_cpu_memoria.json').read_text())
    services = selection['servicios']
    full = Path(os.environ.get('CHRONOGRAPH_DATASET','/microservices-data'))/'node_features.json'
    expected = '907407a33cb5b59ddc276185e2a78ef869f41c6f7a919a0c340fc20467de5e5a'
    actual = sha(full); assert actual == expected
    found = set()
    for service, data in stream_members(full):
        if service not in services: continue
        for resource in RESOURCES:
            assert data[resource] == cohort[service][resource], (service, resource)
        found.add(service)
        if len(found) == len(services): break
    assert found == set(services)
    times = np.array(cohort[services[0]][RESOURCES[0]]['steps'], dtype=np.float64)
    values = np.stack([np.stack([cohort[s][r]['values'] for r in RESOURCES], -1) for s in services], 1)
    assert values.shape == (8005, 28, 2) and np.isfinite(values).all()
    for s in services:
        for r in RESOURCES: assert np.array_equal(times, cohort[s][r]['steps'])
    assert np.all(np.diff(times) > 0)
    edges = list(csv.DictReader(open(Path(os.environ.get('CHRONOGRAPH_DATASET','/microservices-data'))/'edges.csv')))
    internal = [e for e in edges if e['source'] in services and e['target'] in services]
    assert sorted((e['source'], e['target']) for e in internal) == sorted((e['source'], e['target']) for e in selection['aristas_internas'])
    adjacency = np.zeros((28, 28), dtype=np.float32)
    for e in internal: adjacency[services.index(e['source']), services.index(e['target'])] = 1
    np.savez_compressed(out / 'data_verified.npz', values=values, times=times, adjacency=adjacency, services=services)
    audit = dict(full_lfs_sha256=actual, full_lfs_bytes=full.stat().st_size,
                 matched_services=services, shape=list(values.shape), internal_edges=len(internal),
                 gaps=[dict(index=int(i+1), seconds=float(times[i+1]-times[i])) for i in np.where(np.diff(times)!=1800)[0]],
                 cohort_sha256=sha(SOURCE/'datos/cohorte_cpu_memoria.json'),
                 source_zip_sha256='recorded_in_run_config', status='verified_against_local_full_dataset')
    (out/'data_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)

if __name__ == '__main__': main()
