"""Per-container inference monitoring proxy. MLflow's original process is preserved.
Only traffic through port 8081 is measured; native 8080 remains for LK10.
No external Python dependencies, no Docker socket, no secret logging.
"""
import collections
import http.server
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BASELINE = 2.4  # fixed synthetic baseline for LK11 demonstration
THRESHOLD = 1.0  # m/s, demonstration threshold, not validated operational limit
WINDOW = 30
MIN_SAMPLES = 20
BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
BACKEND = os.environ.get('LK11_BACKEND_URL', 'http://127.0.0.1:8080')
PORT = int(os.environ.get('LK11_PROXY_PORT', '8081'))
lock = threading.Lock()
counts = collections.Counter()
buckets = [0] * len(BUCKETS)
duration_sum = 0.0
samples = collections.deque(maxlen=WINDOW)


def record(status, duration, body, response):
    global duration_sum
    observation = None
    if 200 <= status < 300:
        try:
            frame = json.loads(body)['dataframe_split']
            rows = frame['data']
            predictions = json.loads(response)['predictions']
            if len(rows) == len(predictions) == 1:
                raw = float(rows[0][frame['columns'].index('forecast_wind_speed_10m')])
                residual = float(predictions[0])
                if math.isfinite(raw) and math.isfinite(residual):
                    observation = (raw, residual, max(0.0, raw + residual))
        except (KeyError, ValueError, TypeError, IndexError):
            pass
    with lock:
        counts[str(status)] += 1
        duration_sum += duration
        for i, bound in enumerate(BUCKETS):
            if duration <= bound:
                buckets[i] += 1
        if observation is not None:
            samples.append(observation)


def resource_metrics():
    # cgroup v2 first, then v1 paths. Omit unavailable values rather than fabricate.
    result = []
    try:
        usage = dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
        result.append(('lk11_container_cpu_seconds_total', float(usage['usage_usec']) / 1e6))
    except (OSError, ValueError, KeyError):
        for path in ('/sys/fs/cgroup/cpuacct/cpuacct.usage', '/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage'):
            try:
                result.append(('lk11_container_cpu_seconds_total', int(Path(path).read_text()) / 1e9)); break
            except (OSError, ValueError):
                pass
    for path in ('/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory/memory.usage_in_bytes'):
        try:
            result.append(('lk11_container_memory_bytes', int(Path(path).read_text()))); break
        except (OSError, ValueError):
            pass
    return result


def metrics():
    with lock:
        c, b, total, s = dict(counts), list(buckets), duration_sum, list(samples)
    n = sum(c.values())
    out = ['# TYPE lk11_inference_requests_total counter']
    # Explicit successful series enables rate even before first traffic.
    for status in sorted(set(c) | {'200'}):
        out.append(f'lk11_inference_requests_total{{status="{status}"}} {c.get(status, 0)}')
    out.append('# TYPE lk11_inference_duration_seconds histogram')
    for bound, count in zip(BUCKETS, b):
        out.append(f'lk11_inference_duration_seconds_bucket{{le="{bound:g}"}} {count}')
    out += [f'lk11_inference_duration_seconds_bucket{{le="+Inf"}} {n}',
            f'lk11_inference_duration_seconds_sum {total}', f'lk11_inference_duration_seconds_count {n}']
    gauges = {'lk11_window_samples': len(s), 'lk11_input_baseline_ms': BASELINE,
              'lk11_drift_threshold_ms': THRESHOLD, 'lk11_drift_ready': int(len(s) >= MIN_SAMPLES)}
    if s:
        raw, residual, corrected = [sum(v)/len(s) for v in zip(*s)]
        delta = abs(raw - BASELINE)
        gauges.update(lk11_input_wind_mean_ms=raw, lk11_prediction_residual_mean_ms=residual,
                      lk11_corrected_wind_mean_ms=corrected, lk11_input_shift_ms=delta)
        if len(s) >= MIN_SAMPLES:
            gauges['lk11_drift_flag'] = int(delta >= THRESHOLD)
    for key, value in gauges.items():
        out += [f'# TYPE {key} gauge', f'{key} {value}']
    for key, value in resource_metrics():
        kind = 'counter' if key.endswith('_total') else 'gauge'
        out += [f'# TYPE {key} {kind}', f'{key} {value}']
    return ('\n'.join(out) + '\n').encode()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def forward(self, method, body=None):
        start = time.monotonic()
        headers = {'Content-Type': self.headers.get('Content-Type', 'application/json')} if body is not None else {}
        req = urllib.request.Request(BACKEND + self.path, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                status, result, ct = r.status, r.read(), r.headers.get('Content-Type', 'application/json')
        except urllib.error.HTTPError as e:
            status, result, ct = e.code, e.read(), e.headers.get('Content-Type', 'application/json')
        except (OSError, ValueError):
            status, result, ct = 503, b'{"error":"MLflow backend unavailable"}', 'application/json'
        if method == 'POST':
            record(status, time.monotonic() - start, body, result)
        self.reply(status, result, ct)

    def do_GET(self):
        if self.path == '/metrics':
            self.reply(200, metrics(), 'text/plain; version=0.0.4; charset=utf-8')
        elif self.path == '/ping':
            self.forward('GET')
        else:
            self.reply(404, b'Not found', 'text/plain')

    def do_POST(self):
        if self.path != '/invocations':
            self.reply(404, b'Not found', 'text/plain'); return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 1_000_000:
                raise ValueError
        except ValueError:
            self.reply(400, b'Invalid content length', 'text/plain'); return
        self.forward('POST', self.rfile.read(length))


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Missing original MLflow image command')
    child = subprocess.Popen(sys.argv[1:], start_new_session=True)
    server = http.server.ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    server.timeout = 0.5
    stopping = threading.Event()
    def stop(*_):
        stopping.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f'[LK11] Metrics proxy listening on {PORT}; native MLflow backend {BACKEND}', flush=True)
    try:
        while not stopping.is_set() and child.poll() is None:
            server.handle_request()
    finally:
        server.server_close()
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
    if not stopping.is_set():
        raise SystemExit(child.returncode or 1)


if __name__ == '__main__':
    main()
