# Source from anywhere:  source env.sh
# Three uv envs: .venv (main: LeWM + probe, torch), .venvs/isaac (Isaac Sim 4.1, py3.10), .venvs/dreamer (DreamerV3, JAX)
export WM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)
export PYTHONPATH=$WM_ROOT/uav_env${PYTHONPATH:+:$PYTHONPATH}     # scenegen is shared by every env
export DV3_ROOT=$WM_ROOT/third_party/dreamerv3
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export PY_MAIN=$WM_ROOT/.venv/bin/python
export PY_ISAAC=$WM_ROOT/.venvs/isaac/bin/python
export PY_DREAMER=$WM_ROOT/.venvs/dreamer/bin/python
