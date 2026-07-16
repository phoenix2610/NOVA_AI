#!/usr/bin/env python3
"""
nova-ctl — NOVA OS command line control tool
Query watchdog, check system status, send TTS messages manually.

Usage:
  nova-ctl status          — full system status
  nova-ctl ram             — RAM snapshot
  nova-ctl say "text"      — speak via TTS daemon
  nova-ctl ping            — heartbeat check all daemons
  nova-ctl eject <name>    — manually eject a component
"""

import sys
import json
import socket

WATCHDOG_SOCKET = "/tmp/nova_watchdog.sock"
TTS_SOCKET      = "/tmp/nova_tts.sock"
STT_SOCKET      = "/tmp/nova_stt.sock"

def send(sock_path: str, msg: dict) -> dict:
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(sock_path)
        s.sendall((json.dumps(msg) + "\n").encode())
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        s.close()
        return json.loads(data.decode().strip())
    except FileNotFoundError:
        return {"status": "error", "message": f"Socket not found: {sock_path} — is the daemon running?"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def fmt_status(r: dict):
    if r.get("status") != "ok":
        print(f"ERROR: {r.get('message')}")
        return

    ram = r.get("ram", {})
    print(f"\n{'─'*50}")
    print(f"  NOVA OS — System Status")
    print(f"{'─'*50}")
    print(f"  RAM      {ram.get('used_gb')}GB used / {ram.get('total_gb')}GB total")
    print(f"  Free     {ram.get('available_gb')}GB available")
    print(f"  Swap     {ram.get('swap_free_gb')}GB free of {ram.get('swap_total_gb')}GB")
    print(f"\n  Services:")
    for svc, alive in r.get("services", {}).items():
        icon = "✓" if alive else "✗"
        print(f"    {icon}  {svc}")
    loaded = r.get("loaded", [])
    print(f"\n  Loaded components: {', '.join(loaded) if loaded else 'none'}")
    print(f"  Timestamp: {r.get('timestamp')}")
    print(f"{'─'*50}\n")

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0].lower()

    if cmd == "status":
        r = send(WATCHDOG_SOCKET, {"action": "status"})
        fmt_status(r)

    elif cmd == "ram":
        r = send(WATCHDOG_SOCKET, {"action": "ram_check"})
        ram = r.get("ram", {})
        print(f"Available: {ram.get('available_gb')}GB / Total: {ram.get('total_gb')}GB")

    elif cmd == "ping":
        for name, path in [("watchdog", WATCHDOG_SOCKET), ("tts", TTS_SOCKET), ("stt", STT_SOCKET)]:
            r = send(path, {"action": "ping"} if name == "watchdog" else {"priority": -1, "text": ""})
            alive = "online" if "error" not in r.get("status", "error") else "OFFLINE"
            print(f"  {name:12} {alive}")

    elif cmd == "say" and len(args) > 1:
        text = " ".join(args[1:])
        r    = send(TTS_SOCKET, {"priority": 1, "text": text})
        print(f"Queued: {text}")

    elif cmd == "eject" and len(args) > 1:
        r = send(WATCHDOG_SOCKET, {"action": "eject", "component": args[1]})
        print(r.get("message", r))

    else:
        print(__doc__)

if __name__ == "__main__":
    main()
