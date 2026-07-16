#!/usr/bin/env bash
# NOVA OS — install_daemons.sh
# Copies daemon scripts and registers systemd user services.
# Run once: bash install_daemons.sh

set -e

NOVA_HOME="/home/tathya/nova"
SERVICE_DIR="$HOME/.config/systemd/user"

echo "[NOVA] Creating directory structure..."
mkdir -p "$NOVA_HOME/daemons"
mkdir -p "$NOVA_HOME/logs"
mkdir -p "$SERVICE_DIR"

echo "[NOVA] Copying daemon scripts..."
cp tts_daemon.py "$NOVA_HOME/daemons/tts_daemon.py"
cp stt_daemon.py "$NOVA_HOME/daemons/stt_daemon.py"
chmod +x "$NOVA_HOME/daemons/tts_daemon.py"
chmod +x "$NOVA_HOME/daemons/stt_daemon.py"

echo "[NOVA] Installing systemd user services..."
cp nova-tts.service "$SERVICE_DIR/nova-tts.service"
cp nova-stt.service "$SERVICE_DIR/nova-stt.service"

echo "[NOVA] Reloading systemd user daemon..."
systemctl --user daemon-reload

echo "[NOVA] Enabling services (start on login)..."
systemctl --user enable nova-tts.service
systemctl --user enable nova-stt.service

echo "[NOVA] Starting services now..."
systemctl --user start nova-tts.service
sleep 2   # give TTS time to claim audio before STT starts
systemctl --user start nova-stt.service

echo ""
echo "[NOVA] Done. Check status with:"
echo "  systemctl --user status nova-tts"
echo "  systemctl --user status nova-stt"
echo ""
echo "  Live logs:"
echo "  journalctl --user -u nova-tts -f"
echo "  journalctl --user -u nova-stt -f"
