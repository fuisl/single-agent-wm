#!/bin/bash
# LeWM memory line on the kinematic stand-in (docs/lewm_memory_review.md §5):
#   parts_extract, parts_heads, parts: decoders + probes of the trained LeWM's encoder / predictor internals
#   m0..m4: the R-LeWM ladder, each trained, probed and memory-tested on data/kin_v1
# Long runs: launch from a frozen copy (bash scripts/run_frozen.sh ...), never edit a script bash is executing.
#   [SIM=kin] bash scripts/lewm_memory.sh [stage ...]
# Each finished stage writes $R/.done_<stage> and is skipped on re-runs (delete the marker to redo it).
set -euo pipefail
cd "${LEWM_ROOT:-$(dirname "$0")/..}"  # run_frozen.sh sets LEWM_ROOT for the frozen copy
source env.sh
SIM=${SIM:-kin}
D=$WM_ROOT/data/${SIM}_v1 R=$WM_ROOT/runs/${SIM}_v1
STAGES=${*:-parts_extract parts_heads parts}
NTRAIN=300

probe_run() {  # probe_run <out> <name=dir ...>: reuses the label cache of the first probe run
  mkdir -p $1 && ln -sfn $R/probe/label_cache $1/label_cache
  $PY_MAIN probe/probe.py --data $D --out "$@"
}

parts_extract() {
  cd lewm
  $PY_MAIN extract.py --run_dir $R/lewm --data $D --max_extract_train $NTRAIN --parts
  $PY_MAIN extract.py --run_dir $R/lewm --data $D --max_extract_train $NTRAIN --parts --random_init
  cd ..
}

parts_heads() {
  cd lewm
  $PY_MAIN train_heads.py --run_dir $R/lewm --data $D --rep cls --steps 6000
  $PY_MAIN train_heads.py --run_dir $R/lewm --data $D --rep patch --steps 6000 --max_episodes 200 --stride 2
  cd ..
}

parts() {  # <= 3 feature sets per probe run, no frame CNN (~7 GB peak; the frame baseline is in runs/kin_v1/probe)
  probe_run $R/probe_lewm_ref --no_frame --feats wm=$R/wm/features lewm_emb=$R/lewm/features_emb  # frame: runs/kin_v1/probe
  probe_run $R/probe_lewm_parts --no_frame --feats lewm_cls=$R/lewm/features_cls \
    lewm_pred_h=$R/lewm/features_pred_h lewm_pred=$R/lewm/features_pred
  probe_run $R/probe_lewm_patch --no_frame --feats lewm_patch=$R/lewm/features_patch \
    lewm_rand_patch=$R/lewm/features_rand_patch
}

memtest_data() {  # twin look-away scenarios for the label-free memory test (CPU)
  $PY_MAIN uav_env/kin_memtest.py --out $WM_ROOT/data/memtest_v1 --scenarios 60 --workers 10
}

collect_v2() {  # memory-rich training data for M3: the same pilot with look-away glances (CPU)
  $PY_MAIN uav_env/kin_collect.py --workers 10 --episodes 1000 --glance_rate 0.015 --out $WM_ROOT/data/${SIM}_v2
}

STEPS=${STEPS:-15000}
train() {  # train <name> <data dir> <hydra overrides...>
  local name=$1 data=$2; shift 2
  (cd lewm && $PY_MAIN train.py data.dir=$data run_dir=$R/$name trainer.max_steps=$STEPS "$@")
}

evaluate() {  # evaluate <name> <context features>: probe data, memory test and probes, all on kin_v1 / memtest_v1
  (cd lewm && $PY_MAIN extract_mem.py --run_dir $R/$1 --data $D --max_extract_train $NTRAIN \
           && $PY_MAIN memtest.py --run_dir $R/$1 --data $WM_ROOT/data/memtest_v1)
  probe_run $R/probe_$1 --no_frame --feats $1_percept=$R/$1/features_percept $1_$2=$R/$1/features_$2
}

memtest_base() { (cd lewm && $PY_MAIN memtest.py --run_dir $R/lewm --data $WM_ROOT/data/memtest_v1); }
m0() { train m0 $D model=tok; evaluate m0 pred; }                                   # pixels-only + proprio target
m1() { train m1 $D model=tok model.spatial=true; evaluate m1 pred; }               # + 16 spatial tokens
m2() { train m2 $D model=belief data.num_steps=32 loader.batch_size=16; evaluate m2 belief; }   # + recurrent belief
m3() { train m3 $WM_ROOT/data/${SIM}_v2 model=belief data.num_steps=32 loader.batch_size=16 blind.prob=0.5
       evaluate m3 belief; }                                                       # + memory-rich data, blind spans
m4() { train m4 $WM_ROOT/data/${SIM}_v2 model=belief model.belief._target_=module.MapBelief data.num_steps=32 \
         loader.batch_size=16 blind.prob=0.5
       evaluate m4 belief; }                                                       # M3 with an ego-motion-warped map belief

viz() {  # pitch figures + episode videos (CPU, so it can run while the GPU trains); VIZ_SETS adds m* probe sets
  $PY_MAIN scripts/viz/latent_figures.py
  $PY_MAIN scripts/viz/render_episode.py --device cpu --png 200,300,400 \
    --probe_sets wm lewm_emb lewm_patch frame ${VIZ_SETS:-}
}

mkdir -p $R
for s in $STAGES; do
  if [ -f $R/.done_$s ]; then echo "=== skip $s (done)"; continue; fi
  echo "=== lewm_memory $s $(date '+%F %T')"
  $s
  touch $R/.done_$s
done
echo LEWM_MEMORY_DONE
