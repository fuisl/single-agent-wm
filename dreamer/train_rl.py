"""Online DreamerV3 training against rl_server.py (runs in the dreamer env).

    python train_rl.py --logdir runs/dreamer_uav
Start rl_server.py first with the same --num_envs and --port; see run_rl.sh.
"""
import argparse
import os
import pathlib
import sys
from functools import partial as bind

DV3 = pathlib.Path(os.environ["DV3_ROOT"])       # set by env.sh
sys.path.insert(0, str(DV3))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import elements
import embodied
import ruamel.yaml as yaml
from dreamerv3 import main as dm
from dreamerv3.agent import Agent

import isaac_uav_env

def make_agent(config, make_env):
    env = make_env(config, 0)
    obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith("log/")}
    act_space = {k: v for k, v in env.act_space.items() if k != "reset"}
    env.close()
    return Agent(obs_space, act_space, elements.Config(
        **config.agent, logdir=config.logdir, seed=config.seed, jax=config.jax,
        batch_size=config.batch_size, batch_length=config.batch_length, replay_context=config.replay_context,
        report_length=config.report_length, replica=0, replicas=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", required=True)
    p.add_argument("--size", default="size50m")
    p.add_argument("--steps", type=float, default=1.5e6)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--train_ratio", type=float, default=64)
    p.add_argument("--replay_size", type=float, default=1e6)
    p.add_argument("--port", type=int, default=6100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    configs = yaml.YAML(typ="safe").load((DV3 / "dreamerv3" / "configs.yaml").read_text())
    config = elements.Config(configs["defaults"]).update(configs[args.size])
    config = config.update({
        "task": "isaacuav_nav", "logdir": args.logdir, "seed": args.seed,
        "jax.prealloc": False, "replay.size": args.replay_size,
        "run.steps": args.steps, "run.envs": args.envs, "run.train_ratio": args.train_ratio,
        "run.debug": False,                       # parallel env processes: required by the server's lock-step
        "run.log_every": 300, "run.report_every": 1800, "run.save_every": 1800,
        "logger.outputs": ["jsonl"],
    })
    logdir = elements.Path(config.logdir)
    logdir.mkdir()
    config.save(logdir / "config.yaml")
    make_env = bind(isaac_uav_env.make_env, port=args.port)


    run_args = elements.Config(
        **config.run, replica=0, replicas=1, logdir=config.logdir, batch_size=config.batch_size,
        batch_length=config.batch_length, report_length=config.report_length, consec_train=config.consec_train,
        consec_report=config.consec_report, replay_context=config.replay_context)
    embodied.run.train(
        bind(make_agent, config, make_env), bind(dm.make_replay, config, "replay"), bind(make_env, config),
        bind(dm.make_stream, config), bind(dm.make_logger, config), run_args)


if __name__ == "__main__":
    main()
