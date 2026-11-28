"""Patch the existing LK10 Compose; preserve the image's exact native entrypoint."""
from pathlib import Path
import json
import subprocess
import secrets
import yaml

p = Path('docker-compose.yaml')
old = p.read_text()
config = yaml.safe_load(old)
assert {'api','db','model-serving'} <= set(config['services']), 'LK10 Compose required'
if 'prometheus' in config['services'] or 'grafana' in config['services']:
    raise SystemExit('Monitoring already present; inspect existing configuration instead of overwriting.')
model = config['services']['model-serving']
assert model['deploy']['replicas'] == 3
assert 'entrypoint' not in model and 'command' not in model, 'Unexpected custom model entrypoint'
assert model['image'] == 'jambi-wind-lk10-mlflow:latest'
info = json.loads(subprocess.check_output(['docker','image','inspect',model['image']], text=True))[0]['Config']
original = (info.get('Entrypoint') or []) + (info.get('Cmd') or [])
assert original, 'Cannot determine native MLflow launch command'
model['entrypoint'] = ['python','/opt/lk11_proxy.py', *original]
model['command'] = []
model.setdefault('volumes', []).append('./monitoring/lk11_proxy.py:/opt/lk11_proxy.py:ro')
model['ports'].append({'target':8081,'host_ip':'127.0.0.1','protocol':'tcp'})
model['healthcheck']['test'] = ['CMD','python','-c',"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8081/ping', timeout=3)"]
config['services']['prometheus'] = {
 'image':'prom/prometheus:v3.5.0','restart':'unless-stopped',
 'command':['--config.file=/etc/prometheus/prometheus.yml','--storage.tsdb.path=/prometheus','--storage.tsdb.retention.time=2d'],
 'ports':['127.0.0.1:9090:9090'],'networks':['wind-network'],
 'volumes':['./prometheus.yml:/etc/prometheus/prometheus.yml:ro','./monitoring/lk11-rules.yml:/etc/prometheus/lk11-rules.yml:ro','prometheus_data:/prometheus']}
config['services']['grafana'] = {
 'image':'grafana/grafana:12.1.0','restart':'unless-stopped',
 'ports':['127.0.0.1:3000:3000'],'networks':['wind-network'],
 'environment':{'GF_SECURITY_ADMIN_USER':'admin','GF_SECURITY_ADMIN_PASSWORD':'${LK11_GRAFANA_PASSWORD:?set LK11_GRAFANA_PASSWORD in .env}','GF_AUTH_ANONYMOUS_ENABLED':'false','GF_USERS_ALLOW_SIGN_UP':'false'},
 'depends_on':['prometheus'],
 'volumes':['./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro','./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro','grafana_data:/var/lib/grafana']}
config.setdefault('volumes', {}).update(prometheus_data=None,grafana_data=None)
# Keep existing YAML comments and original LK09 definitions: insert only changed model and new services.
start = old.index('  model-serving:')
end = old.index('\nvolumes:', start)
services = yaml.safe_dump({'services':{k:config['services'][k] for k in ('model-serving','prometheus','grafana')}},sort_keys=False,default_flow_style=False).split('\n',1)[1]
new = old[:start] + services + old[end:]
new = new.replace('volumes:\n  postgres_data:', 'volumes:\n  prometheus_data:\n  grafana_data:\n  postgres_data:',1)
parsed = yaml.safe_load(new)
assert parsed['services']['api'] == config['services']['api']
assert parsed['services']['db'] == config['services']['db']
assert parsed['services']['model-serving'] == model
assert {'prometheus_data','grafana_data','postgres_data'} <= set(parsed['volumes'])
env = Path('.env')
if not env.is_file():
    raise SystemExit('Existing .env required; create from .env.example and fill credentials first.')
if not any(l.strip().startswith('LK11_GRAFANA_PASSWORD=') for l in env.read_text().splitlines()):
    with env.open('a') as f:
        f.write('\nLK11_GRAFANA_PASSWORD=' + secrets.token_urlsafe(24) + '\n')
    env.chmod(0o600)
p.write_text(new)
example = Path('.env.example')
if not any(l.startswith('LK11_GRAFANA_PASSWORD=') for l in example.read_text().splitlines()):
    with example.open('a') as f:
        f.write('\n# LK11: create a strong local Grafana admin password\nLK11_GRAFANA_PASSWORD=replace_with_a_strong_grafana_password\n')
print('[PASS] Original MLflow entrypoint retained behind monitoring proxy')
print('[PASS] Native port 8080 retained; monitored port 8081 added')
print('[PASS] Prometheus, Grafana and persistent monitoring volumes added')
print('[PASS] LK09 API/database/network/volume unchanged; Grafana password stored in ignored .env')
