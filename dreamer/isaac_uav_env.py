"""DreamerV3 (embodied) env that is one drone slot of rl_server.py. Runs in the dreamer env.

The connection is opened lazily on the first step, so creating the env only to read its spaces
(as DreamerV3's make_agent does) does not take a slot.
"""
import time
from multiprocessing.connection import Client

import elements
import embodied
import numpy as np


class IsaacUAV(embodied.Env):

    def __init__(self, index, port=6100, authkey="airdreamer", reset_options=None):
        self.index, self.port, self.authkey = index, port, authkey
        self.reset_options = reset_options or {}      # e.g. {"kind": "utrap", "seed": 7} for evaluation
        self._conn = None
        self._done = True

    @property
    def obs_space(self):
        spaces = {
            "depth": elements.Space(np.uint8, (48, 80, 1)),
            "state": elements.Space(np.float32, (15,)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            "log/goal": elements.Space(np.float32),
            "log/collision": elements.Space(np.float32),
            "log/height": elements.Space(np.float32),
            "log/timeout": elements.Space(np.float32),
        }
        return spaces

    @property
    def act_space(self):
        return {"action": elements.Space(np.float32, (5,), -1.0, 1.0), "reset": elements.Space(bool)}

    def step(self, action):
        if self._conn is None:
            for attempt in range(60):                 # the server may still be busy accepting other slots
                try:
                    self._conn = Client(("127.0.0.1", self.port), authkey=self.authkey.encode())
                    break
                except (ConnectionRefusedError, ConnectionResetError, EOFError):
                    time.sleep(1.0)
            else:
                raise ConnectionError(f"slot {self.index}: cannot reach rl_server on port {self.port}")
            self._conn.send(("hello", self.index))
        if action["reset"] or self._done:
            self._conn.send(("reset", self.next_reset_options()))
        else:
            self._conn.send(("step", np.asarray(action["action"], np.float32)))
        obs = self._conn.recv()
        obs.pop("pos", None); obs.pop("yaw", None)          # evaluation-only extras
        self._done = bool(obs["is_last"])
        return obs

    def next_reset_options(self):
        return self.reset_options

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def make_env(config, index, port=6100, **overrides):
    """Picklable env factory for DreamerV3's parallel driver."""
    from dreamerv3.main import wrap_env
    return wrap_env(IsaacUAV(index, port=port, **overrides), config)
