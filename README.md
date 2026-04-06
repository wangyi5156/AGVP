# AGVP

## 📋 Environment Requirements
This project is developed with Python 3.7 on Ubuntu 24.04. If you are using miniconda or anaconda, you can create an environment with following instructions.

```bash
conda create -n agvp python=3.7 cmake=3.14.0 -y
conda activate agvp
```

#### Install dependences (habitat-lab/habitat-sim)

```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout RLRAudioPropagationUpdate
python setup.py install --headless --audio

git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.2.2
pip install -e .
```

##### Edit habitat/tasks/rearrange/rearrange_sim.py file and remove the 36th line where FetchRobot is imported.

#### Install soundspaces

```bash
git clone https://github.com/facebookresearch/sound-spaces.git
cd sound-spaces
pip install -e .
```

#### Download scene datasets
```bash
cd sound-spaces
mkdir data && cd data
mkdir scene_datasets && cd scene_datasets
```

### 1.Training
```bash
python ss_baselines/av_nav/run.py --exp-config ss_baselines/av_nav/config/audionav/replica/train_telephone/audiogoal_depth.yaml --model-dir data/models/replica/audiogoal_depth
```

### 2.Validation (evaluate each checkpoint and generate a validation curve)
```bahs
python ss_baselines/av_nav/run.py --run-type eval --exp-config ss_baselines/av_nav/config/audionav/replica/val_telephone/audiogoal_depth.yaml --model-dir data/models/replica/audiogoal_depth
```

### 3.Test the best validation checkpoint based on validation curve

```bash
python ss_baselines/av_nav/run.py --run-type eval --exp-config ss_baselines/av_nav/config/audionav/replica/test_telephone/audiogoal_depth.yaml --model-dir data/models/replica/audiogoal_depth EVAL_CKPT_PATH_DIR data/models/replica/audiogoal_depth/data/ckpt.XXX.pth
```
