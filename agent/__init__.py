from .base import BaseAgent
from .rl_agent import RLAgent

# Always-available baselines
from .maxpressure import MaxPressureAgent
from .fixedtime import FixedTimeAgent
from .sotl import SOTLAgent

# Optional RL agents; import lazily so missing deps (e.g., torch_scatter) don't block baselines
try:
	from .colight import CoLightAgent
except ModuleNotFoundError:
	CoLightAgent = None

try:
	from .dqn import DQNAgent
except ModuleNotFoundError:
	DQNAgent = None

try:
	from .frap import FRAP_DQNAgent
except ModuleNotFoundError:
	FRAP_DQNAgent = None

try:
	from .ppo_pfrl import IPPO_pfrl
except ModuleNotFoundError:
	IPPO_pfrl = None

# from .maddpg import MADDPGAgent
try:
	from .maddpg_v2 import MADDPGAgent
except ModuleNotFoundError:
	MADDPGAgent = None

try:
	from .magd import MAGDAgent
except ModuleNotFoundError:
	MAGDAgent = None

try:
	from .presslight import PressLightAgent
except ModuleNotFoundError:
	PressLightAgent = None

try:
	from .mplight import MPLightAgent
except ModuleNotFoundError:
	MPLightAgent = None
