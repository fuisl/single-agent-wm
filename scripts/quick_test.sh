#!/bin/bash
# End-to-end smoke test on tiny data, every stage of the pipeline once. Stops at the first failure.
#   bash scripts/quick_test.sh [stage ...]     stages: isaac collect dreamer lewm probe online
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
D=$WM_ROOT/data/wm_test R=$WM_ROOT/runs/test
STAGES=${*:-isaac collect dreamer lewm probe online}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.6}

isaac() {
  (cd $R && $PY_ISAAC $WM_ROOT/uav_env/smoke_test.py)
}

collect() {
  $PY_ISAAC uav_env/collect.py --num_envs 4 --episodes 40 --max_steps 400 --out $D --validate
}

dreamer() {
  $PY_DREAMER dreamer/train_wm.py train   --data $D --logdir $R/wm --size size12m --steps 400
  $PY_DREAMER dreamer/train_wm.py extract --data $D --logdir $R/wm --size size12m
  $PY_DREAMER dreamer/train_wm.py extract --data $D --logdir $R/wm --size size12m --random_init --feat_dir features_rand
}

lewm() {
  cd lewm
  $PY_MAIN train.py data.dir=$D run_dir=$R/lewm trainer.max_steps=400 trainer.warmup_steps=50 trainer.log_every=50 \
    trainer.val_every=100
  $PY_MAIN train_heads.py --run_dir $R/lewm --data $D --steps 1000
  $PY_MAIN extract.py --run_dir $R/lewm --data $D
  $PY_MAIN extract.py --run_dir $R/lewm --data $D --random_init
  cd ..
}

probe() {
  $PY_MAIN probe/probe.py --data $D --out $R/probe --epochs 2 --feats wm=$R/wm/features rand=$R/wm/features_rand \
    lewm=$R/lewm/features_ctx lewm_emb=$R/lewm/features_emb lewm_rand=$R/lewm/features_rand_ctx
}

serve() {  # serve <envs> <port> <log>: rl_server with --extras for evaluation; exits when its clients disconnect
  $PY_ISAAC uav_env/rl_server.py --num_envs $1 --port $2 --extras > $3 2>&1 &
  until grep -q SERVER_READY $3; do kill -0 $! || { tail -20 $3; return 1; }; sleep 5; done
}

online() {  # short DreamerV3 online run, then both controllers on two test suites
  ENVS=4 PORT=6150 bash dreamer/run_rl.sh $R/rl --size size12m --steps 3000 --train_ratio 32
  serve 4 6151 $R/eval_dreamer_server.log
  $PY_DREAMER dreamer/eval_rl.py --logdir $R/rl --port 6151 --envs 4 --episodes 4 --suites random,utrap_in
  serve 4 6152 $R/eval_lewm_server.log
  (cd lewm && $PY_MAIN eval.py run_dir=$R/lewm port=6152 envs=4 episodes=4 suites=random,utrap_in solver.n_steps=5)
}

mkdir -p $R
for s in $STAGES; do echo "=== quick_test $s $(date +%T)"; $s; done
echo QUICK_TEST_DONE
