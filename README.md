# Incorporating Multistep Prediction and Value into DreamerV2
This is a project for CSC413: Neural Networks and Deep Learning at the University of Toronto. We investigate different methods to incorporate value into the DreamerV2 model, which performed well on long-term reward environments in RL. This repository is forked from the implementation of the DreamerV2 agent in TensorFlow 2, whose code can be found [here](https://github.com/danijar/dreamerv2).

## Contributions
Our project paper highlighting the details of this work can be downloaded here [PDF Upload Link]. Our main contributions (modifications) to the existing architecture are the following: <Br>
**Contribution 1- Value-Aware Dreamer:** We strived to train world models that are informed by the value
- We encode projected value into a hidden state during training by exploiting the connected computation graphs between the World Model and Actor-Critic Module.
- In considering the literature on the value based reinforcement learning framework, we integrate a theoretically supported value-aware world model learning method, iterVAML, into the Dreamer architecture (Iterative Value-Aware Model Learning, Farahmand A, 2018)

**Contribution 2- Multistep Prediction:** We hypothesized that representations which embody information about the future are more apt for future planning
- We work to integrate multi step awareness into the hidden states by encoding training individual hidden states to be reconstructable into several observations into the future.

## Reproducing Results
### Installation
Perhaps the easiest way to install all necessary packages would be to build a conda/virtualenv environment from our yaml/txt file, which was used on our VMs to run experiments. 
#### Conda
```
conda env create -f dreamerenv.yml	# create & install
conda activate dreamerenv			      # activate
```
#### Virtualenv
```
python3 -m venv myenv				      # create
source myenv/bin/activate			    # activate
pip3 install -r requirements.txt	# install
```
**Note** that our virtual environment(s) are entirely based on the [manual installation instructions](https://github.com/danijar/dreamerv2?tab=readme-ov-file#manual-instructions) from the original repo so feel free to follow those. We just found it to be a difficult process and these env files made the process simple for setting up on VMs.
### Training
Train (from `valuedream/` directory). The default log directory is `valuedream/logs/`
```sh

python3  dreamerv2/train.py  --logdir  /logdir/1

```
As per our ablation study (isolating different components) you can add any combination of the following three flags. Note however that the `itervaml` flag will force `wm_backpropvalue` to be true as it is required.
`--itervaml True --wm_backpropvalue True --multistep True`
 
You can monitor results with 
```
tensorboard --logdir logdir
```
 ### Metrics
You can plot the results using the following script. Modify the `STATS_DIR` and `METHODS` list in `plotall.py` to decide which results you want displayed after training.
```sh
python3 plotting/plotall.py
```

### Results
![Returns over time](https://github.com/mishaalkandapath/valuedream/blob/main/plots/reward.pdf)
![Score over time](https://github.com/mishaalkandapath/valuedream/blob/main/plots/scores_time.pdf)
![Spectrum of achievements at 1M steps](https://github.com/mishaalkandapath/valuedream/blob/main/plots/spectrum-reward.pdf)
