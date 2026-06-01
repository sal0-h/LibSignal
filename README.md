# Introduction

![visitors](https://visitor-badge.laobi.icu/badge?page_id=wingsweihua.LibSignal&style=flat)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fdarl-libsignal.github.io%2F&up_message=LibSignal&style=flat)](https://darl-libsignal.github.io/)
![GitHub Repo stars](https://img.shields.io/github/stars/sal0-h/LibSignal?style=flat&color=red)

> **Note:** This repository continues [DaRL-LibSignal/LibSignal](https://github.com/DaRL-LibSignal/LibSignal) with Python 3.11+ compatibility, setup improvements, and ongoing maintenance. Use this repo for installs and experiments; the original upstream is largely inactive.

This repo provides OpenAI Gym-compatible environments for traffic light control scenarios and a bunch of baseline methods. 

Environments include single intersections (single-agent) and multi-intersections (multi-agents) with different road networks and traffic flow settings.

Baselines include traditional Traffic Signal Control algorithms and reinforcement learning-based methods.

LibSignal is a cross-simulator environment that provides multiple traditional and Reinforcement Learning models in traffic control tasks. Currently, we support SUMO, CityFlow, and CBEine simulation environments. Conversion between SUMO and CityFlow is carefully calibrated.

## 🚀 🚀 🚀
## We have created a docker image for your convenience 
## <span style="color:red">(Run LibSignal, multiple sim2real baselines by one line)!</span>


This docker code base contains three projects, first pull from docker hub: 

`docker pull danielda1/ugat:latest`

`docker run -it --name ugat_case danielda1/ugat:latest`

For LibSignal - Then go to the terminal: 

`cd /DaRL/LibSignal`

`python run.py`

We have also included two sim-to-real for RL - TSC tasks:  

> CDC23: Uncertainty-aware Grounded Action Transformation towards Sim-to-Real Transfer for Traffic Signal Control (https://github.com/darl-libsignal/ugat)

`cd /DaRL/UGAT_Docker/`

`python sim2real.py`

> AAAI24: Prompt to Transfer: Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning (https://github.com/DaRL-LibSignal/PromptGAT)

`cd /DaRL/PromptGAT`

`python sim2real.py`

# Install

This fork is developed and tested with **SUMO** (`--world sumo`). CityFlow code paths remain in the tree but are not actively maintained here.

## Quick setup (recommended)

```bash
git clone https://github.com/sal0-h/LibSignal.git
cd LibSignal
chmod +x setup.sh
./setup.sh
```

`setup.sh` creates the conda env `traffic` (Python 3.10), installs PyTorch (CUDA if available), SUMO 1.26 (`libsumo` / `traci`), `torch-geometric`, `torch-scatter` (for CoLight), and the Python packages in `requirements.txt`.

On a shared server where system packages are already installed:

```bash
./setup.sh --no-sudo
```

Activate before running experiments:

```bash
conda activate traffic
python run.py --task tsc --agent presslight --world sumo --network sumo1x1 --prefix test
```

## Manual / partial install

If you already have conda and CUDA, you can install pip dependencies after PyTorch and SUMO:

```bash
pip install -r requirements.txt
```

CoLight also needs `torch-scatter` (the setup script installs it via conda-forge). See [team_instructions.pdf](./team_instructions.pdf) for Colab and lab-server workflows.

## Optional: CityFlow

Upstream LibSignal supports `--world cityflow` if [CityFlow](https://github.com/cityflow-project/CityFlow) is installed. We do not test that path in this fork today; use SUMO for team experiments. A CityFlow ↔ SUMO converter lives in [common/converter.py](./common/converter.py).

## Agents

RL agents are imported automatically from `agent/__init__.py` when dependencies are present (e.g. CoLight needs `torch_scatter` + `torch_geometric`). Baselines (`maxpressure`, `fixedtime`, `sotl`) work without those extras.
# Start

## Run Model Pipeline

Our library has a uniform structure that empowers users to start their experiments with just one click. Users can start an experiment by setting arguments in the run.py file and start with their customized settings. The following part is the arguments provided to customize.

```
python run.py
```

Supporting parameters:

- <font color=red> thread_num:  </font> number of threads for cityflow simulation

- <font color=red> ngpu:  </font> how many gpu resources used in this experiment

- <font color=red> task:  </font> task type to run

- <font color=red> agent:  </font> agent type of agents in RL environment

- <font color=red> world:  </font> simulator type

- <font color=red> dataset:  </font> type of dataset in training process

- <font color=red> path:  </font> path to configuration file

- <font color=red> prefix:  </font> the number of predix in this running process

- <font color=red> seed:  </font> seed for pytorch backend
  </br></br>


# Maintaining plan

*<font size=4>To ensure the stability of our traffic signal testbed, we will first push new code onto **dev** branch, after validation, then merge it into the master branch. </font>*

| **UPdate index**           | **Date**      | **Status** | **Merged** |
|----------------------------|---------------|------------|------------|
| **MPLight implementation** | July-18-2022  | developed  | √          |
| **Libsumo integration**    | August-8-2022 | developed | √          |
| **Delay calculation**      | August-8-2022 | developed |  √          |
| **CoLight adaptation for heterogenous network** | November-2-2024 | developed | √  |
| **Optimize FRAP and MPLight**      | October-4-2022 | developed |  √          |
| **FRAP adaptation for irregular intersections**      | October-18-2022 | developed |  √          |
| **PettingZoo envrionment to better support MARL**      | Jul-18-2023 | developed |       |
| **RLFX Agent controlling phase and duration**      | Jul-18-2023 | developed |    |
| **Ray rllib support**      | Jul-18-2023 | developling |   |

# Citation

LibSignal is accepted by the Machine Learning Journal by Springer: ```Mei, H., Lei, X., Da, L. et al. Libsignal: an open library for traffic signal control. Mach Learn (2023). https://doi.org/10.1007/s10994-023-06412-y``` and can be cited with the following BibTeX entry (A short version is accepted by NeurIPS 2022 Workshop: Reinforcement Learning for Real Life):

```
@article{mei2023libsignal,
  title={Libsignal: an open library for traffic signal control},
  author={Mei, Hao and Lei, Xiaoliang and Da, Longchao and Shi, Bin and Wei, Hua},
  journal={Machine Learning},
  pages={1--37},
  year={2023},
  publisher={Springer}
}
```
