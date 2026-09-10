#!/usr/bin/env bash
# Waits for the running VGG-19 CIFAR-100 @90 job to exit, then launches
# ResNet-32 CIFAR-100 @90 fully detached (setsid+nohup, stdin from /dev/null).
set -u
JENKS_DIR="/home/jvap2/Desktop/Code/TIME_ML/Python_Jenks_Test/Jenks_Tests"
WAITPID="${1:-3496105}"          # VGG C100 @90 python PID
cd "$JENKS_DIR" || exit 1

echo "[queue] waiting on PID $WAITPID (VGG C100 @90) at $(date)"
while kill -0 "$WAITPID" 2>/dev/null; do
    sleep 60
done
echo "[queue] PID $WAITPID exited at $(date); launching ResNet-32 C100 @90"

# small settle so the GPU frees fully
sleep 30

TS=$(date +%Y%m%d_%H%M%S)
LOG="$JENKS_DIR/run_logs/resnet32_c100_90_${TS}.log"
setsid nohup /usr/bin/python3 -u "$JENKS_DIR/ResNet32_CIFAR100_90.py" > "$LOG" 2>&1 < /dev/null &
echo "[queue] launched ResNet-32 C100 @90 -> $LOG (pid $!)"
