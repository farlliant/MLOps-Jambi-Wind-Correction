"""Read Prometheus instant query results, validate scrape coverage and drift state."""
import argparse
import json
import time
import urllib.parse
import urllib.request

BASE='http://127.0.0.1:9090'
def query(expr):
    with urllib.request.urlopen(BASE+'/api/v1/query?'+urllib.parse.urlencode({'query':expr}),timeout=10) as r:
        x=json.load(r)
    if x['status']!='success': raise RuntimeError(x)
    return x['data']['result']

def main():
    p=argparse.ArgumentParser();p.add_argument('--expect',choices=['targets','baseline','shift','recovery'],default='targets');a=p.parse_args()
    deadline=time.monotonic()+90
    while True:
        try:
            up=query('up{job="mlflow-model"}')
            target_ok=len(up)==3 and all(float(x['value'][1])==1 for x in up)
            flags=query('lk11_drift_flag{job="mlflow-model"}')
            cpu=query('lk11_container_cpu_seconds_total{job="mlflow-model"}')
            ram=query('lk11_container_memory_bytes{job="mlflow-model"}')
            resources_ok=len(cpu)==len(ram)==3
            if a.expect=='targets': valid=target_ok and resources_ok
            else:
                wanted=1 if a.expect=='shift' else 0
                valid=target_ok and resources_ok and len(flags)==3 and all(float(x['value'][1])==wanted for x in flags)
            if valid: break
        except (OSError,ValueError,KeyError,RuntimeError):
            valid=False
        if time.monotonic()>=deadline: raise SystemExit('[FAIL] Prometheus targets/resource/drift checks not satisfied; inspect /targets and metrics')
        time.sleep(5)
    expressions={'targets':'up{job="mlflow-model"}','requests':'sum by(instance)(lk11_inference_requests_total{job="mlflow-model"})','throughput_rps':'sum by(instance)(rate(lk11_inference_requests_total{job="mlflow-model"}[30s]))','latency_p95_seconds':'histogram_quantile(0.95,sum by(instance,le)(rate(lk11_inference_duration_seconds_bucket{job="mlflow-model"}[30s])))','cpu_seconds':'lk11_container_cpu_seconds_total{job="mlflow-model"}','ram_bytes':'lk11_container_memory_bytes{job="mlflow-model"}','window_samples':'lk11_window_samples{job="mlflow-model"}','raw_mean_ms':'lk11_input_wind_mean_ms{job="mlflow-model"}','residual_mean_ms':'lk11_prediction_residual_mean_ms{job="mlflow-model"}','corrected_mean_ms':'lk11_corrected_wind_mean_ms{job="mlflow-model"}','input_shift_ms':'lk11_input_shift_ms{job="mlflow-model"}','drift_flags':'lk11_drift_flag{job="mlflow-model"}','alerts':'ALERTS{alertname="SyntheticInputDrift"}'}
    for k,e in expressions.items():
        print(k+': '+json.dumps([{'instance':r['metric'].get('instance'),'value':r['value'][1],**({'state':r['metric']['alertstate']} if 'alertstate' in r['metric'] else {})} for r in query(e)]))
    print('[PASS] Prometheus verification:',a.expect)
if __name__=='__main__':main()
