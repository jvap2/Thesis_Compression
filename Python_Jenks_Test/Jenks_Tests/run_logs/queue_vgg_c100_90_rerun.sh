#!/usr/bin/env bash
# Waits for the running ResNet-32 CIFAR-100 @90 job to exit, then launches the
# VGG-19 CIFAR-100 @90 RERUN (now OVER_PRUNE=0.30) fully detached.
set -u
JENKS_DIR="/home/jvap2/Desktop/Code/TIME_ML/Python_Jenks_Test/Jenks_Tests"
WAITPID="${1:?need ResNet python PID}"     # ResNet C100 @90 python PID
cd "$JENKS_DIR" || exit 1

echo "[queue] waiting on PID $WAITPID (ResNet-32 C100 @90) at $(date)"
while kill -0 "$WAITPID" 2>/dev/null; do
    sleep 60
done
echo "[queue] PID $WAITPID exited at $(date); launching VGG-19 C100 @90 rerun"
sleep 30   # let GPU free

TS=$(date +%Y%m%d_%H%M%S)
LOG="$JENKS_DIR/run_logs/vgg19_c100_90_${TS}.log"
setsid nohup /usr/bin/python3 -u "$JENKS_DIR/VGG19_CIFAR100_90.py" > "$LOG" 2>&1 < /dev/null &
echo "[queue] launched VGG-19 C100 @90 rerun -> $LOG (wrapper pid $!)"
