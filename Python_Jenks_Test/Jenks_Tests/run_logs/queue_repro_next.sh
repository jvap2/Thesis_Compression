#!/usr/bin/env bash
# Waits for the current OP=0.10/LR5e-2 repro run to exit, then launches the next
# repro run (LR 4e-2, WD 2e-4, OP 0.10, prune 225, total 350, batch 300, LS 0.1).
set -u
JD="/home/jvap2/Desktop/Code/TIME_ML/Python_Jenks_Test/Jenks_Tests"
WAITPID="${1:?need current python PID}"
cd "$JD" || exit 1
echo "[queue] waiting on PID $WAITPID at $(date)"
while kill -0 "$WAITPID" 2>/dev/null; do sleep 30; done
echo "[queue] PID $WAITPID exited at $(date); launching next repro run"
sleep 30
TS=$(date +%Y%m%d_%H%M%S)
LOG="$JD/run_logs/vgg19_c100_90_repro_op10_lr4e2_pe225_${TS}.log"
setsid nohup /usr/bin/python3 -u "$JD/VGG19_CIFAR100_90_repro7354.py" > "$LOG" 2>&1 < /dev/null &
echo "[queue] launched -> $LOG (wrapper pid $!)"
