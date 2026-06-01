# LibSignal

[Website](https://darl-libsignal.github.io/)
GitHub Repo stars

OpenAI Gymnasium-compatible environments for **traffic signal control (TSC)** with classical and reinforcement-learning baselines.

**Maintained at:** [sal0-h/LibSignal](https://github.com/sal0-h/LibSignal) — standalone project with Python 3.10+ tooling, SUMO-focused workflows, and team/server setup (`setup.sh`, [team_instructions.pdf](./team_instructions.pdf)).

### Upstream LibSignal (please cite)

This codebase is based on the open-source **[LibSignal](https://github.com/DaRL-LibSignal/LibSignal)** library by the DaRL group ([project site](https://darl-libsignal.github.io/)). We gratefully use their environments, baselines, and simulator integrations; **academic work should cite the original publication** (see [Citation](#citation) below), not only this maintained copy.


|                         |                                                                                                                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Original repository** | [https://github.com/DaRL-LibSignal/LibSignal](https://github.com/DaRL-LibSignal/LibSignal)                                                                                     |
| **Paper**               | Mei, H. et al., *Libsignal: an open library for traffic signal control*, Machine Learning (2023). [doi:10.1007/s10994-023-06412-y](https://doi.org/10.1007/s10994-023-06412-y) |


The upstream repository is largely inactive; use **this repo** for installs and day-to-day experiments.

Environments cover single- and multi-intersection networks. Baselines include MaxPressure, fixed-time, SOTL, DQN, PressLight, CoLight, MPLight, and others.

**Simulator focus here:** SUMO (`--world sumo`). CityFlow/OpenEngine paths exist in the codebase from upstream but are not actively tested in this repo.

## 🚀 🚀 🚀

## We have created a docker image for your convenience

## (Run LibSignal, multiple sim2real baselines by one line)!

This docker code base contains three projects, first pull from docker hub: 

`docker pull danielda1/ugat:latest`

`docker run -it --name ugat_case danielda1/ugat:latest`

For LibSignal - Then go to the terminal: 

`cd /DaRL/LibSignal`

`python run.py`

We have also included two sim-to-real for RL - TSC tasks:  

> CDC23: Uncertainty-aware Grounded Action Transformation towards Sim-to-Real Transfer for Traffic Signal Control ([https://github.com/darl-libsignal/ugat](https://github.com/darl-libsignal/ugat))

`cd /DaRL/UGAT_Docker/`

`python sim2real.py`

> AAAI24: Prompt to Transfer: Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning ([https://github.com/DaRL-LibSignal/PromptGAT](https://github.com/DaRL-LibSignal/PromptGAT))

`cd /DaRL/PromptGAT`

`python sim2real.py`

# Install

Developed and tested with **SUMO** (`--world sumo`).

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

Upstream LibSignal supports `--world cityflow` if [CityFlow](https://github.com/cityflow-project/CityFlow) is installed. We do not test that path here; use SUMO for experiments. A CityFlow ↔ SUMO converter lives in [common/converter.py](./common/converter.py).

## Agents

RL agents are imported automatically from `agent/__init__.py` when dependencies are present (e.g. CoLight needs `torch_scatter` + `torch_geometric`). Baselines (`maxpressure`, `fixedtime`, `sotl`) work without those extras.

# Start

## Run Model Pipeline

Our library has a uniform structure that empowers users to start their experiments with just one click. Users can start an experiment by setting arguments in the run.py file and start with their customized settings. The following part is the arguments provided to customize.

```
python run.py
```

Supporting parameters:

- thread_num: number of threads for cityflow simulation
- ngpu: how many gpu resources used in this experiment
- task: task type to run
- agent: agent type of agents in RL environment
- world: simulator type
- dataset: type of dataset in training process
- path: path to configuration file
- prefix: the number of predix in this running process
- seed: seed for pytorch backend

# Citation

If you use LibSignal (including this maintained repository) in research, **cite the original LibSignal paper and reference the upstream repository**:

- **Paper:** Mei, H., Lei, X., Da, L. et al. Libsignal: an open library for traffic signal control. *Machine Learning* (2023). [https://doi.org/10.1007/s10994-023-06412-y](https://doi.org/10.1007/s10994-023-06412-y)  
- **Code (original):** [https://github.com/DaRL-LibSignal/LibSignal](https://github.com/DaRL-LibSignal/LibSignal)

A short version was also presented at the NeurIPS 2022 Workshop *Reinforcement Learning for Real Life*.

```bibtex
@article{mei2023libsignal,
  title={Libsignal: an open library for traffic signal control},
  author={Mei, Hao and Lei, Xiaoliang and Da, Longchao and Shi, Bin and Wei, Hua},
  journal={Machine Learning},
  pages={1--37},
  year={2023},
  publisher={Springer},
  doi={10.1007/s10994-023-06412-y}
}
```

