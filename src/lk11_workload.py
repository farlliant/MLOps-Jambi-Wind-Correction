"""Send actual synthetic inference requests concurrently to all three monitored replicas."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import random
import subprocess
import time
import urllib.request
from train import FEATURES
from src.lk07_inference import build_synthetic_input


def endpoint(index):
    s = subprocess.check_output(['docker','compose','port','--index',str(index),'model-serving','8081'],text=True).strip().splitlines()[-1]
    return 'http://127.0.0.1:' + s.rsplit(':',1)[-1]


def run(index, url, seconds, mean, seed):
    rng = random.Random(seed + index)
    frame, _, _ = build_synthetic_input(FEATURES)
    deadline = time.monotonic() + seconds
    ok = failed = 0
    while time.monotonic() < deadline:
        start = time.monotonic()
        raw = max(0, rng.uniform(mean - .3, mean + .3))
        frame.loc[0,'forecast_wind_speed_10m'] = raw
        frame.loc[0,'forecast_wind_gusts_10m'] = raw + 1.7
        payload = {'dataframe_split':frame.to_dict(orient='split')}
        req = urllib.request.Request(url+'/invocations',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(req,timeout=15) as res:
                pred = json.load(res)['predictions']
                if res.status != 200 or len(pred)!=1 or not math.isfinite(float(pred[0])):
                    raise ValueError('invalid response')
                ok += 1
        except (OSError, ValueError, KeyError, IndexError) as exc:
            failed += 1
            if failed <= 3:
                print(f'[WARN] replica {index}: {type(exc).__name__}',flush=True)
        time.sleep(max(0, 1.0-(time.monotonic()-start)))
    return {'replica':index,'endpoint':url,'success':ok,'failed':failed}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['baseline','shift','recovery'],required=True)
    p.add_argument('--seconds',type=int,default=75)
    a=p.parse_args()
    if a.seconds < 30: p.error('Use at least 30 seconds to fill the rolling window')
    mean=5.4 if a.phase=='shift' else 2.4
    urls=[endpoint(i) for i in (1,2,3)]
    print(f'[INFO] {a.phase}: synthetic wind mean={mean} m/s, duration={a.seconds}s, 3 concurrent clients',flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(run,i,u,a.seconds,mean,100 if a.phase!='shift' else 200) for i,u in enumerate(urls,1)]
        results=[f.result() for f in fs]
    print(json.dumps(results,indent=2),flush=True)
    if any(x['failed'] or x['success']<30 for x in results):
        raise SystemExit('[FAIL] Workload incomplete; investigate before claiming success')
    print('[PASS] Actual inference requests succeeded on all three replicas')

if __name__=='__main__': main()
