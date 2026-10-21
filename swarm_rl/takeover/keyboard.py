"""Loopback-only browser keyboard takeover with real-time simulation and logs.

Serve on the research server, then SSH-forward port 8765 to the Mac. No person
is simulated here: takeover events and velocity commands must come from UI.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time

import numpy as np
import torch

from .adaptation import ObservationHistory, load_policy, make_env
from .adaptation_train import EpisodeMeter, autonomous_mask, line, write


PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>Quad-Swarm 人工接管预实验</title>
<style>body{font:16px system-ui;background:#101927;color:#e5ecf6;max-width:1100px;margin:24px auto}button{padding:10px;margin:5px;background:#243c58;color:white;border:1px solid #7590ad;border-radius:6px}canvas{background:#18283b;border-radius:12px;width:100%}#status{white-space:pre-wrap;line-height:1.6}b{color:#ffbd69}</style>
<h2>四无人机 · 人工接管预实验</h2>
<p>初始暂停。先点“继续”，按 <b>T 接管橙色无人机</b>，按 <b>G 交还</b>。</p>
<p>按住 W/S：前后（x）；A/D：左右（y）；Q/E：升降（z）。P 暂停，R 新一轮。松键或窗口失焦后速度指令归零。</p>
<button onclick="send('pause')">暂停 / 继续 P</button><button onclick="send('takeover')">接管 T</button><button onclick="send('handback')">交还 G</button><button onclick="send('reset')">新一轮 R</button>
<p>橙色：本轮可接管无人机；青色：自主队友；空心圆：当前巡航槽位；十字：人工巡检目标。左：俯视 XY，右：侧视 XZ。</p>
<canvas id="view" width="1100" height="420"></canvas><p id="status"></p><p id="error" style="color:#ff8c8c"></p>
<script>
const token='SESSION_TOKEN', held=new Set(), canvas=document.getElementById('view'), ctx=canvas.getContext('2d');
let busy=false;
async function send(event){try{const r=await fetch('/input',{method:'POST',headers:{'Content-Type':'application/json','X-Session':token},body:JSON.stringify({event,keys:[...held]})});if(!r.ok)throw Error(await r.text());document.getElementById('error').textContent='';}catch(e){document.getElementById('error').textContent=String(e)}}
const events={t:'takeover',g:'handback',p:'pause',r:'reset'};
onkeydown=e=>{let k=e.key.toLowerCase();if(events[k]&&!e.repeat)send(events[k]);if('wasdqe'.includes(k)&&k.length===1){held.add(k);e.preventDefault()}};
onkeyup=e=>held.delete(e.key.toLowerCase());onblur=()=>{held.clear();send(null)};
function draw(s){ctx.clearRect(0,0,1100,420);let mean=s.positions.reduce((a,p)=>a+p[0],0)/4;
for(let panel=0;panel<2;panel++){let axis=panel===0?1:2;function point(p){return [275+panel*550+(p[0]-mean)*65,210-(p[axis]-(panel===0?0:2.5))*65]}
ctx.strokeStyle='#2b435c';for(let a=0;a<550;a+=65){ctx.beginPath();ctx.moveTo(panel*550+a,0);ctx.lineTo(panel*550+a,420);ctx.stroke()}ctx.fillStyle='#a8bdd3';ctx.fillText(panel===0?'XY 俯视':'XZ 侧视',panel*550+15,22);
s.positions.forEach((p,i)=>{let [x,y]=point(p),[tx,ty]=point(s.slots[i]);ctx.strokeStyle='#8495aa';ctx.beginPath();ctx.arc(tx,ty,10,0,7);ctx.stroke();ctx.fillStyle=i===s.agent?'#ffbd69':'#5ed8e3';ctx.beginPath();ctx.arc(x,y,8,0,7);ctx.fill();ctx.fillText('UAV '+i,x+11,y-8)});
if(s.target){let [x,y]=point(s.target);ctx.strokeStyle='#ffbd69';ctx.beginPath();ctx.moveTo(x-9,y);ctx.lineTo(x+9,y);ctx.moveTo(x,y-9);ctx.lineTo(x,y+9);ctx.stroke()}}
document.getElementById('status').textContent=`轮次 ${s.episode} | t=${s.time.toFixed(1)} s | ${s.paused?'暂停':'运行'} | 阶段 ${s.phase}\n队友策略 ${s.policy} | 碰撞 ${s.collision} | 巡检完成 ${s.check_completed} | 恢复成功 ${s.recovery_success}\n${s.done?'本轮结束，按 R 新建；未交还的轮次保留为未完成。':''}`}
setInterval(async()=>{if(busy)return;busy=true;try{await send(null);let r=await fetch('/state');draw(await r.json())}catch(e){document.getElementById('error').textContent=String(e)}finally{busy=false}},100);
</script></html>'''


class Session:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.token = secrets.token_urlsafe(24)
        self.lock = threading.RLock()
        self.events = (self.output / 'events.jsonl').open('w', buffering=1)
        self.trajectory = (self.output / 'trajectory.jsonl').open('w', buffering=1)
        self.model, self.saved = load_policy(args.checkpoint) if args.checkpoint else (None, None)
        self.variant = self.saved['variant'] if self.saved else 'nominal'
        self.frames = self.saved['frames'] if self.saved else 4
        self.episode = -1
        self.last_input = 0
        self.keys = set()
        self.reset()
        write(self.output / 'session.json', {'mode': 'actual_keyboard_operator',
            'geometry': args.geometry, 'policy': self.variant, 'checkpoint': args.checkpoint,
            'seed_start': args.seed, 'limits': 'single-operator feasibility only; no population claim'})

    def record_end(self, reason):
        write(self.output / f'episode-{self.episode:04d}.json', {'reason': reason,
            'episode': self.episode, 'seed': self.args.seed + self.episode,
            'complete_handoff_cycle': self.env.handover_time is not None,
            'episode_ended': bool(self.env.terminated or self.env.truncated), **self.meter.result(self.env)})

    def reset(self):
        if self.episode >= 0:
            self.record_end('user_reset')
        self.episode += 1
        self.env = make_env(self.args.geometry, 'steady', manual=True)
        raw, _ = self.env.reset(seed=self.args.seed + self.episode)
        self.history = ObservationHistory(self.variant if self.model else 'flag', self.frames)
        self.obs = self.history.reset(raw)
        self.meter = EpisodeMeter()
        self.paused, self.keys = True, set()
        line(self.events, {'event': 'reset', 'episode': self.episode, 'seed': self.args.seed + self.episode})

    def input(self, payload):
        with self.lock:
            self.last_input = time.monotonic()
            self.keys = set(payload.get('keys', [])) & set('wasdqe')
            event = payload.get('event')
            if event == 'reset':
                self.reset()
            elif event == 'pause':
                self.paused = not self.paused
            elif event == 'takeover':
                self.env.begin_takeover()
                self.history.buffer.popleft()
                self.obs = self.history.push(self.env.observe())
            elif event == 'handback':
                if self.env.terminated or self.env.truncated:
                    raise ValueError('Episode ended; start a new episode')
                self.env.force_handback()
                # Authority change replaces the current observation, not a fake
                # extra elapsed-time history frame.
                self.history.buffer.popleft()
                self.obs = self.history.push(self.env.observe())
            elif event is not None:
                raise ValueError('Unknown event')
            if event:
                line(self.events, {'event': event, 'episode': self.episode,
                    'simulation_time': self.env.time, 'phase': self.env.phase})

    def state(self):
        with self.lock:
            e = self.env
            return {'episode': self.episode, 'time': e.time, 'paused': self.paused,
                'phase': e.phase, 'agent': e.takeover_agent, 'positions': e.positions.tolist(),
                'slots': e.slot_targets.tolist(), 'target': None if e._operator_target is None else e._operator_target.tolist(),
                'policy': self.variant, 'done': bool(e.terminated or e.truncated),
                'collision': bool(e.collision), 'check_completed': bool(e.check_completed),
                'recovery_success': bool(e.recovery_success)}

    def loop(self):
        try:
            while True:
                started = time.monotonic()
                with self.lock:
                    # Loss of browser heartbeat pauses simulation instead of
                    # letting a disconnected operator remain responsible.
                    if self.last_input and started - self.last_input > 0.5:
                        self.keys.clear()
                        self.paused = True
                    if not self.paused and not (self.env.terminated or self.env.truncated):
                        keys = self.keys if started - self.last_input < 0.35 else set()
                        self.env.manual_velocity = 0.7 * np.array([
                            int('w' in keys) - int('s' in keys), int('a' in keys) - int('d' in keys),
                            int('q' in keys) - int('e' in keys)], dtype=float)
                        if self.model:
                            with torch.no_grad():
                                action = self.model.deterministic(torch.from_numpy(self.obs), autonomous_mask(self.env)).numpy()
                        else:
                            action = np.zeros((4, 3), dtype=np.float32)
                        self.meter.before(self.env)
                        raw, _, _, _, _ = self.env.step(action)
                        self.meter.after(self.env)
                        self.obs = self.history.push(raw)
                        line(self.trajectory, {'episode': self.episode, 'keys': sorted(keys),
                            'actions': action.tolist(), 'pre_phase': self.meter.pre_phase,
                            'manual_velocity': self.env.manual_velocity.tolist(),
                            'requested_velocity': self.env.last_requested_velocity.tolist(),
                            'executed_velocity': self.env.last_executed_velocity.tolist(),
                            'override_mask': self.env.last_override_mask.tolist(), **self.state()})
                        if self.env.terminated or self.env.truncated:
                            self.record_end('episode_end')
                time.sleep(max(0.0, 0.1 - (time.monotonic() - started)))
        except Exception as exc:
            write(self.output / 'server_error.json', {'error': repr(exc)})
            self.paused = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--geometry', choices=('outward', 'crossing'), default='crossing')
    parser.add_argument('--checkpoint')
    parser.add_argument('--seed', type=int, default=60000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    session = Session(args)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, body, content_type):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body.encode())

        def do_GET(self):
            if self.path == '/':
                self.respond(200, PAGE.replace('SESSION_TOKEN', session.token), 'text/html; charset=utf-8')
            elif self.path == '/state':
                self.respond(200, json.dumps(session.state(), allow_nan=False), 'application/json')
            else:
                self.respond(404, 'Not found', 'text/plain')

        def do_POST(self):
            if self.path != '/input' or self.headers.get('X-Session') != session.token:
                self.respond(403, 'Invalid session', 'text/plain')
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size < 4096:
                    raise ValueError('Invalid payload size')
                session.input(json.loads(self.rfile.read(size)))
                self.respond(200, '{}', 'application/json')
            except (ValueError, TypeError) as exc:
                self.respond(400, str(exc), 'text/plain')

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    threading.Thread(target=session.loop, daemon=True).start()
    print(f'Keyboard UI: http://127.0.0.1:{args.port} (loopback only, initially paused)', flush=True)
    try:
        server.serve_forever()
    finally:
        with session.lock:
            session.record_end('server_stop')


if __name__ == '__main__':
    main()
