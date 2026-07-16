#!/usr/bin/env python3
"""Quick smoke test for the NOVA executioner."""

import json
import socket

SOCKET = "/tmp/nova_executioner.sock"

def send(payload: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(SOCKET)
        s.sendall((json.dumps(payload) + "\n").encode())
        raw = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            raw += chunk
            if raw.endswith(b"\n"):
                break
        return json.loads(raw.decode())

tests = [
    {
        "name": "Tier 1 — system stats",
        "payload": {
            "action": "execute",
            "operations": ["system_stats"],
            "script": "import psutil\nprint(f'CPU: {psutil.cpu_percent()}%  RAM: {psutil.virtual_memory().percent}%')"
        }
    },
    {
        "name": "Tier 1 — audio volume check",
        "payload": {
            "action": "execute",
            "operations": ["audio_control"],
            "script": "import subprocess\nr = subprocess.run(['wpctl','get-volume','@DEFAULT_AUDIO_SINK@'],capture_output=True,text=True)\nprint(r.stdout.strip())"
        }
    },
    {
        "name": "Tier 1 — file write to workspace",
        "payload": {
            "action": "execute",
            "operations": ["write_workspace"],
            "script": "import os\npath = os.path.expanduser('~/nova/workspace/test_exec.txt')\nwith open(path,'w') as f: f.write('executioner ok')\nprint(f'Written: {path}')"
        }
    },
    {
        "name": "Banned op — should be blocked",
        "payload": {
            "action": "execute",
            "operations": ["write_workspace"],
            "script": "import os\nos.system('rm -rf /')"
        }
    }
]

print("=" * 60)
print("NOVA Executioner — Smoke Tests")
print("=" * 60)

for t in tests:
    print(f"\n▶ {t['name']}")
    try:
        result = send(t["payload"])
        if result.get("blocked"):
            print(f"  ✓ BLOCKED as expected: {result.get('error')}")
        elif result.get("success"):
            print(f"  ✓ OK ({result.get('duration_ms')}ms)")
            if result.get("stdout"):
                print(f"    → {result['stdout']}")
        else:
            print(f"  ✗ FAILED: {result.get('stderr') or result.get('error')}")
    except Exception as e:
        print(f"  ✗ CONNECTION ERROR: {e}")

print("\n" + "=" * 60)
