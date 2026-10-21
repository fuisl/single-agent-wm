#!/bin/bash
# Full run: collect -> LeWM -> DreamerV3 WM -> probe -> LeWM CEM control eval -> DreamerV3 online RL + eval.
#   [SIM=kin] bash scripts/pipeline.sh [stage ...]
# Each finished stage writes $R/.done_<stage> and is skipped on re-runs (delete the marker to redo it).
# Stops at the first failure; fix and re-run the same command to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
SIM=${SIM:-isaac}
D=$WM_ROOT/data/${SIM}_v1 R=$WM_ROOT/runs/${SIM}_v1
if [ $SIM = kin ]; then SERVER=kin_server.py PY_SERVER=$PY_MAIN; else SERVER=rl_server.py PY_SERVER=$PY_ISAAC; fi
export SERVER XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.7}
STAGES=${*:-collect lewm dreamer_wm probe lewm_eval dreamer_rl dreamer_eval}
WM_SIZE=${WM_SIZE:-size12m}    # earlier results used size50m (OOM on 6 GB); 12m ~ LeWM's 18M params, 2 h / 20k steps
NTRAIN=300                     # train episodes exported for probe fitting (RAM); all val episodes are exported

collect() {
  if [ $SIM = kin ]; then
    $PY_MAIN uav_env/kin_collect.py --workers 12 --episodes 1000 --out $D
  else
    $PY_ISAAC uav_env/collect.py --num_envs 8 --episodes 1000 --out $D
  fi
}

lewm() {
  cd lewm
  $PY_MAIN train.py data.dir=$D run_dir=$R/lewm trainer.max_steps=30000
  $PY_MAIN train_heads.py --run_dir $R/lewm --data $D --steps 10000
  $PY_MAIN extract.py --run_dir $R/lewm --data $D --max_extract_train $NTRAIN
  $PY_MAIN extract.py --run_dir $R/lewm --data $D --random_init --max_extract_train $NTRAIN
  cd ..
}

dreamer_wm() {
  $PY_DREAMER dreamer/train_wm.py train   --data $D --logdir $R/wm --size $WM_SIZE --steps 20000
  $PY_DREAMER dreamer/train_wm.py extract --data $D --logdir $R/wm --size $WM_SIZE --max_extract_train $NTRAIN
  $PY_DREAMER dreamer/train_wm.py extract --data $D --logdir $R/wm --size $WM_SIZE --random_init --feat_dir features_rand --max_extract_train $NTRAIN
}

probe() {  # two runs to fit in 14 GB RAM: trained models, then untrained baselines; the frame CNN is in both
  $PY_MAIN probe/probe.py --data $D --out $R/probe \
    --feats wm=$R/wm/features lewm=$R/lewm/features_ctx lewm_emb=$R/lewm/features_emb
  $PY_MAIN probe/probe.py --data $D --out $R/probe_untrained \
    --feats lewm_rand=$R/lewm/features_rand_ctx rand=$R/wm/features_rand
}

serve() {  # serve <envs> <port> <log>: env server with --extras; exits when its clients disconnect
  $PY_SERVER uav_env/$SERVER --num_envs $1 --port $2 --extras > $3 2>&1 &
  until grep -q SERVER_READY $3; do kill -0 $! || { tail -20 $3; return 1; }; sleep 5; done
}

lewm_eval() {
  serve 16 6160 $R/lewm/eval_server.log
  (cd lewm && $PY_MAIN eval.py run_dir=$R/lewm port=6160 envs=16 episodes=100 \
    solver.num_samples=128 solver.n_steps=10 solver.topk=16)   # 300 / 30 / 30 is ~10x slower on this GPU
}

dreamer_rl() {
  ENVS=16 PORT=6170 bash dreamer/run_rl.sh $R/rl --size $WM_SIZE --steps ${RL_STEPS:-5e5}   # earlier results: 1.5e6 (~9 h here)
}

dreamer_eval() {
  serve 16 6180 $R/rl/eval_server.log
  $PY_DREAMER dreamer/eval_rl.py --logdir $R/rl --port 6180 --envs 16 --episodes 100
}

mkdir -p $R
for s in $STAGES; do
  if [ -f $R/.done_$s ]; then echo "=== skip $s (done)"; continue; fi
  echo "=== pipeline $s $(date '+%F %T')"
  $s
  touch $R/.done_$s
done
echo PIPELINE_DONE
