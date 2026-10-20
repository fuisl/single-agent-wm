#!/bin/bash
# Usage: bash dreamer/run_rl.sh <logdir> [extra train_rl.py args]
# Starts the Isaac env server, waits until it listens, then runs DreamerV3 training against it.
# Env vars: ENVS (drones, default 16), PORT (6100), MEMFRAC (JAX GPU memory fraction, 0.5),
#           SERVER_ARGS (extra rl_server.py args).
source "$(dirname "$0")/../env.sh"
LOGDIR=$(realpath -m $1); shift
PORT=${PORT:-6100}
mkdir -p $LOGDIR
$PY_ISAAC $WM_ROOT/uav_env/rl_server.py --num_envs ${ENVS:-16} --port $PORT ${SERVER_ARGS:-} > $LOGDIR/server.log 2>&1 &
SERVER=$!
trap "kill $SERVER 2>/dev/null" EXIT
until grep -q SERVER_READY $LOGDIR/server.log; do
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -20 $LOGDIR/server.log; exit 1; }
  sleep 5
done
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.5}
$PY_DREAMER $WM_ROOT/dreamer/train_rl.py --logdir $LOGDIR --envs ${ENVS:-16} --port $PORT "$@" 2>&1 | tee $LOGDIR/train.log
