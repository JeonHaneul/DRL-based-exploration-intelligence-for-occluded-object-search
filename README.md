# DRL-based Exploration Intelligence for Occluded Object Search

## Paper

This repository provides the code for the following research paper:

**A study on deep reinforcement learning-based exploration intelligence for occluded object search**  
*Engineering Applications of Artificial Intelligence* (2026)  
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0952197626012364) | [DOI: 10.1016/j.engappai.2026.114954](https://doi.org/10.1016/j.engappai.2026.114954)

This repository contains the training and experiment code together in a single repository under `src/`.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/DRL_training` | DRL training and evaluation code |
| `src/DRL_experiment` | ROS 2 experiment and robot-control code |

## System requirements and environment setup

### Reference workstation

| Component | Reference specification | Notes |
| --- | --- | --- |
| CPU | AMD Ryzen 9 5950X | scene generation, simulation orchestration, ROS 2 build |
| RAM | 128 GB | large Isaac Sim scenes, parallel environments and datasets |
| GPU | NVIDIA RTX 3090 / 3090 Ti / 4090 | **24 GB VRAM is the key requirement** for RGB rendering, FCN and vectorized DRL training |
| Storage | NVMe SSD recommended | Isaac Sim cache, USD assets, 24,000+ images and checkpoints require substantial space |
| OS | Ubuntu 22.04 recommended | matches ROS 2 Humble and the project’s Linux paths/scripts |

### Training environment (Isaac Sim 4.2)

The training pipeline was developed for **Isaac Sim 4.2** with a compatible Isaac Lab checkout. The code uses the legacy `omni.isaac.lab` namespace, so do not replace it with an arbitrary current Isaac Lab release without migration.

1. Install an NVIDIA driver supported by Isaac Sim 4.2 and confirm `nvidia-smi` detects the 24 GB GPU.
2. Install Isaac Sim 4.2 and the repository’s pinned Isaac Lab submodule.
3. Clone the repository with submodules and configure long paths if cloning on Windows.

```bash
git clone --recurse-submodules https://github.com/JeonHaneul/DRL-based-exploration-intelligence-for-occluded-object-search.git
cd DRL-based-exploration-intelligence-for-occluded-object-search
git submodule update --init --recursive
```

4. Install the training-side Python dependencies inside the Isaac Lab Python environment. The checked requirements snapshot includes PyTorch 2.4.0 + CUDA 11.8, torchvision 0.19.0, NumPy 1.26.0, OpenCV 4.9, Pillow 10.2, SciPy 1.10, Matplotlib 3.8, Gymnasium 0.29, RSL-RL and TensorBoard.

```bash
cd src/DRL_training/IsaacLab
./isaaclab.sh -i
./isaaclab.sh -p -m pip install \
  numpy==1.26.0 pillow==10.2.0 opencv-python==4.9.0.80 \
  scipy==1.10.1 matplotlib==3.8.4 tqdm==4.67.1
```

> `src/DRL_training/requirements.txt` is a full environment snapshot containing ROS packages, editable Git dependencies and machine-specific build URLs. Do **not** blindly install the entire file into Isaac Sim. Start from Isaac Sim 4.2/Isaac Lab, then install only missing packages. Install PyTorch explicitly only for a separate standalone FCN virtual environment; avoid overwriting Isaac Sim’s bundled torch unless the CUDA/Python compatibility has been verified.

5. Configure the Omniverse/Nucleus USD paths used by `environment.yaml`, scene generators and DirectRLEnv. Update hard-coded `MODEL_PATH`, dataset output paths and camera/shelf geometry before running.

### Real-world experiment environment

The real system uses an Intel RealSense D455 RGB-D camera, a UR5e manipulator, a Robotiq 2F-85 gripper, a front-access Speedrack shelf and an auxiliary placing zone. The software workspace targets **Ubuntu 22.04 + ROS 2 Humble**.

```bash
sudo apt update
sudo apt install ros-humble-desktop python3-colcon-common-extensions python3-rosdep
source /opt/ros/humble/setup.bash
sudo rosdep init  # first installation only
rosdep update
rosdep install --from-paths src/DRL_experiment/src --ignore-src -r -y

cd src/DRL_experiment
colcon build --symlink-install --allow-overriding ur_description
source install/setup.bash
```

Before operating hardware:

- Install the RealSense SDK/ROS wrapper and verify aligned RGB, depth and point cloud topics from the D455.
- Configure a static IP between the workstation and UR5e, enable Remote Control, install/activate the External Control URCap, and verify the UR ROS 2 driver before loading MoveIt.
- Configure the Robotiq 2F-85 serial/USB permissions and confirm the gripper driver/controller independently.
- Calibrate the D455-to-robot transform. The experiment memo uses `world -> camera1_link` static TF and the paper places the camera about 85 cm from the shelf.
- Test camera, perception, FCN/DRL service and robot motion separately before launching the integrated sequence.
- Keep an emergency stop available and validate motions at reduced speed in a cleared workspace.

External open-source dependencies such as Isaac Lab, ViSP, MoveIt messages, Web Video Server, Robotiq, and Serial remain Git submodules. Clone them with:

```bash
git submodule update --init --recursive
```

---

## DRL_training README

The following section mirrors `src/DRL_training/README.md`.

# DRL Training

## 학습 파이프라인 개요

학습은 다음 순서로 진행합니다.

1. `src/high_level_dataset`에서 FCN 학습용 RGB 장면, 타깃 ground truth, 최종 distribution map을 생성합니다.
2. `FCN_model_training`에서 데이터 이름과 폴더 구조를 통일하고 mean/std를 계산한 뒤 FCN-ResNet50을 학습·검증합니다.
3. 생성된 FCN weight를 `src/High_level_policy_direct`의 DirectRLEnv에 연결하고 RSL-RL PPO로 DRL action selector를 학습합니다.

> **현재 저장소 상태:** FCN 학습 코드는 `src/DRL_training/src/FCN_model_training`에 포함되어 있으며 실제 파일명은 `rename.py`, `find_config.py`, `train_250506.py`, `test_pred.py`입니다. 아래 명령은 이 실제 경로와 파일명을 기준으로 합니다.

## 1. FCN 학습 데이터 생성

데이터 생성 코드는 [`src/high_level_dataset`](./src/high_level_dataset)에 있습니다. 모든 스크립트는 Isaac Sim/Isaac Lab Python으로 실행해야 하며, Omniverse의 shelf/object USD asset 경로가 코드의 `omniverse://localhost/Library/Shelf/...` 경로와 일치해야 합니다.

### 1.1 `Scene_generator.py`: 선반 장면과 scene mask 생성

[`Scene_generator.py`](./src/high_level_dataset/Scene_generator.py)는 타깃 및 주변 객체를 무작위로 배치하고 RGB, depth, semantic/instance segmentation과 column별 scene mask를 저장합니다. 코드의 `target_row_index`, `spawn_probability`, `visibility_probability`, `ENV_Cfg.shelf`, `usd_path_mapping`을 실험 구성에 맞게 먼저 확인하십시오.

```bash
./isaaclab.sh -p source/standalone/shelf_env/Scene_generator.py \
  --target_object can_2 \
  --enable_camera \
  --save \
  --num_img 100
```

| 인자 | 기본값 | 의미 |
| --- | ---: | --- |
| `--target_object` | `cup_1` | 생성할 타깃 객체 이름입니다. USD mapping과 클래스 목록에 존재해야 합니다. |
| `--num_img` | `10` | 생성할 장면/프레임 수입니다. 코드에서는 정수로 변환해 사용합니다. |
| `--save` | 꺼짐 | RGB, depth, segmentation, mask 출력을 디스크에 저장합니다. |
| `--camera_id` | `0` | 저장·시각화할 카메라 인덱스이며 허용값은 0 또는 1입니다. |
| `--draw` | 꺼짐 | GUI에서 point cloud marker를 표시합니다. |
| `--num_envs` | `2` | App/scene 생성용 환경 수 인자입니다. 현재 스크립트의 실제 데이터 생성 루프에서 사용 여부를 확인하십시오. |
| AppLauncher 인자 | - | `--headless`, camera enable 등 Isaac Lab이 추가하는 인자입니다. 버전에 따라 `--enable_camera` 대신 `--enable_cameras`일 수 있으므로 `--help`로 확인하십시오. |

출력은 코드 파일 기준의 `output/camera/<target_object>/scene/` 아래에 생성됩니다.

### 1.2 `Target_scene_gnerator.py`: 타깃 배치 가능 영역 생성

[`Target_scene_gnerator.py`](./src/high_level_dataset/Target_scene_gnerator.py)는 타깃 객체를 선반의 가능한 위치와 회전으로 이동시키면서 target RGB/depth/semantic 데이터를 생성합니다. 파일명은 원본의 `gnerator` 철자를 그대로 사용합니다.

```bash
./isaaclab.sh -p source/standalone/shelf_env/Target_scene_gnerator.py \
  --target_object can_3 \
  --enable_camera \
  --save \
  --row 4
```

| 인자 | 기본값 | 의미 |
| --- | ---: | --- |
| `--target_object` | `cup_1` | ground-truth 위치를 생성할 타깃 클래스입니다. |
| `--row` | `1` | 생성 범위를 제한하는 선반 row 설정입니다. shelf 좌표 간격 및 종료 조건과 함께 맞춰야 합니다. |
| `--save` | 꺼짐 | target camera 데이터를 저장합니다. |
| `--camera_id` | `0` | 사용할 카메라 인덱스(0 또는 1)입니다. |
| `--draw` | 꺼짐 | GUI point cloud 표시를 활성화합니다. |
| AppLauncher 인자 | - | headless/camera 관련 Isaac Lab 공통 인자입니다. |

출력은 `output/camera/<target_object>/target/` 아래에 생성됩니다. `usd_path_mapping`, shelf 원점, camera pose와 `scene_update()`의 이동 간격은 실제 선반 규격에 맞춰야 합니다.

### 1.3 `Final_distribution_map_generator.py`: 최종 FCN label 생성

[`Final_distribution_map_generator.py`](./src/high_level_dataset/Final_distribution_map_generator.py)는 scene/target semantic segmentation과 depth를 결합해 occlusion distribution을 계산하고 similarity mask와 합성하여 최종 distribution map을 만듭니다.

```bash
./isaaclab.sh -p source/standalone/shelf_env/Final_distribution_map_generator.py \
  --target_object can_2 \
  --save
```

| 인자/설정 | 기본값 | 의미 |
| --- | ---: | --- |
| `--target_object` | `cup_1` | 처리할 타깃 클래스입니다. scene/target 폴더 이름과 일치해야 합니다. |
| `--save` | 꺼짐 | 계산된 depth distribution map을 저장합니다. |
| `folder_path` | 코드 내 절대 경로 | `__main__`의 경로를 실제 `output/camera` 위치로 수정해야 합니다. |
| `occlusion_threshold` | `0.25` | target 영역 중 occluded로 판정할 최소 비율입니다. |
| 결합 비율 | 코드 확인 필요 | 제공된 최종 스크립트는 일부 구간에서 depth/similarity를 0.2/0.8로 결합합니다. 논문 Eq. (2)의 재현 목표는 similarity 가중치 `β=0.7`입니다. |

예상 입력/출력 폴더는 `semantic_segmentation`, `distance_to_camera`, `processed_depth`, `depth_dis_map`, `mask`, `distribution_map`입니다. 각 scene과 target의 파일 개수와 번호가 일치해야 합니다.

## 2. FCN 모델 학습

### 2.1 파일 이름 통일: `rename.py`

`rename.py`는 RGB 이름(`rgb_<number>_0.png`)과 label 이름(`01_<number>.png`)을 동일한 7자리 번호(`0000001.png`)로 복사해 `train_x/<class>`, `train_y/<class>` 구조를 만듭니다.

```bash
cd src/DRL_training/src/FCN_model_training
python rename.py
```

CLI 인자는 없으므로 파일 상단의 다음 값을 클래스마다 수정해 반복 실행합니다.

- `rgb_folder`, `masks_folder`: 원본 RGB와 distribution map 폴더
- `train_x_folder`, `train_y_folder`: 정리된 입력/label 저장 폴더
- 클래스 이름(예: `bottle_4`): 두 입력과 두 출력 경로에서 동일하게 유지

### 2.2 normalization 계산: `find_config.py`

`find_config.py`는 전체 `train_x` 이미지의 RGB mean/std와 최소 height/width를 계산하고 `outputs/dataset_stats.txt`, `outputs/global_min_size.txt`에 저장합니다.

```bash
python find_config.py
```

CLI 인자는 없습니다. `train_x_dir`, `class_names`, `output_dir`을 데이터셋과 일치시키고, 데이터가 많으면 `calculate_mean_std(..., batch_size=1000)`의 batch size를 메모리에 맞게 조정합니다.

### 2.3 FCN-ResNet50 학습: `train.py`

현재 저장소의 실제 학습 파일명은 `train_250506.py`입니다.

```bash
python train_250506.py
python train_250506.py --resume
# 원본 이름을 유지한 경우:
# python train_250506.py [--resume]
```

`--resume`은 `Config.MODEL_PATH`의 weight를 불러와 이어서 학습합니다. 주요 수정 지점은 다음과 같습니다.

| Config | 검토 코드 값 | 의미 |
| --- | ---: | --- |
| `X_DATA_DIR`, `Y_DATA_DIR` | `./train_x`, `./train_y` | 입력 RGB와 distribution label root |
| `OUTPUT_DIR` | `./outputs` | weight, log, 예측 이미지 저장 위치 |
| `BATCH_SIZE` | `16` | 640x480 입력에서 24 GB VRAM 기준 batch |
| `LR` | `1e-4` | Adam learning rate |
| `EPOCHS` | `100` | 최대 epoch |
| `CLASS_NAMES` | 16 classes | 폴더명, output channel 순서와 반드시 동일 |
| `LEARNING_DATA_RATIO` | `0.8` | train/validation split |
| `MODEL_PATH` | `outputs/best_model_45.pth` | best/resume weight |
| `SAVE_IMAGES_INTERVAL` | `2` | validation image와 epoch weight 저장 간격 |

모델은 pretrained FCN-ResNet50의 classifier를 클래스 수만큼 변경하고, class별 output channel에 MSE Loss를 적용합니다. optimizer는 Adam이며 normalization은 `dataset_stats.txt`를 읽습니다. 논문 Table 5 기준은 batch 16, learning rate 0.0001, 100 epochs, MSE Loss, Adam, early stopping입니다.

### 2.4 FCN 테스트: `test.py`

현재 저장소의 실제 테스트 파일명은 `test_pred.py`입니다.

```bash
python test_pred.py --target can_2
# 원본 이름을 유지한 경우:
# python test_pred.py --target can_2
```

`--target`은 `CLASS_NAMES` 중 추론할 클래스입니다. `TEST_DIR`, `OUTPUT_DIR`, `MODEL_PATH`, `DATASET_STATS_PATH`, `BATCH_SIZE`, `CLASS_NAMES`를 학습 구성과 동일하게 맞춰야 합니다.

## 3. FCN weight를 이용한 DRL 학습

### 3.1 DirectRLEnv 코드 위치

- 환경/config/FCN 추론: [`src/High_level_policy_direct/high_level_policy_direct_env.py`](./src/High_level_policy_direct/high_level_policy_direct_env.py)
- RSL-RL PPO 설정: [`src/High_level_policy_direct/agents/rsl_rl_cfg.py`](./src/High_level_policy_direct/agents/rsl_rl_cfg.py)
- Gym task 등록: [`src/High_level_policy_direct/__init__.py`](./src/High_level_policy_direct/__init__.py)

`HighlevelDirectEnvCfg`에서 `scene.num_envs`, `decimation`, `episode_length_s`, action/observation space, camera, RGB `mean/std`, `MODEL_PATH`를 설정합니다. `HighlevelDirectEnv.__init__()`은 FCN-ResNet50을 만들고 `MODEL_PATH`의 weight를 로드합니다. `_get_observations()`은 RGB normalization, FCN inference, gain `g=2`, column별 1D-PDM, temporal smoothing(논문 `α=0.7`; 현재 코드 변수명 `gamma`)을 계산하는 위치입니다.

> **구현 확인 필요:** 현재 GitHub의 `high_level_policy_direct_env.py`는 `_get_rewards()`가 `torch.zeros(...)`를 반환하고 action/observation도 테스트용 스텁 상태입니다. 아래 논문 보상은 설명만 추가한 것이며, 논문의 완전한 Direct 환경 구현에서 config 상수와 `_get_rewards()`로 이식하지 않으면 실제 PPO 학습에 보상이 전달되지 않습니다.

### 3.2 논문 reward/penalty와 weight

논문 Eqs. (12)-(24), Table 4의 값입니다. full Direct 구현에서는 weight를 `HighlevelDirectEnvCfg`의 상수로 두고, 조건 계산과 합산은 `HighlevelDirectEnv._get_rewards()`에 둡니다.

| 항목 | 조건 요약 | weight |
| --- | --- | ---: |
| Target grasping reward | 선택 column의 target을 grasp | `+60` |
| Sweeping right reward | Top-2 column, right 공간 존재, rightmost 제외 | `+35` |
| Sweeping left reward | Top-2 column, left 공간 존재, leftmost 제외 | `+35` |
| Non-target grasping reward | Top-2이며 양쪽 sweep이 어려울 때 non-target 제거 | `+5` |
| Low-probability grasp penalty | Top-2가 아닌 column grasp | `-5` |
| Low-probability sweep-right penalty | Top-2가 아닌 column sweep right | `-5` |
| Low-probability sweep-left penalty | Top-2가 아닌 column sweep left | `-5` |
| Empty-column action penalty | 선택 column에 객체 없음 | `-10` |
| Target sweeping penalty | target을 좌/우로 sweep | `-20` |
| Grasp-when-sweep-possible penalty | sweep 공간이 있는데 grasp 선택 | `-20` |
| Re-sweep penalty | 직전 sweep을 반대로 되돌림 | `-15` |
| Grasp-previously-swept penalty | 직전 sweep한 객체를 바로 grasp | `-10` |
| Termination penalty | collision 또는 workspace/shelf 이탈 | `-15` |

논문은 3x4 shelf에서 observation 15개(4-column PDM, 4 depth, 4 object ID, target ID, 이전 action 2개)와 12개 action(3 primitives x 4 columns)을 사용합니다. 4x5 transfer 시 observation 18개와 15개 action으로 확장합니다.

### 3.3 RSL-RL PPO 설정과 실행

현재 [`rsl_rl_cfg.py`](./src/High_level_policy_direct/agents/rsl_rl_cfg.py)의 주요 값은 다음과 같습니다.

| 설정 | 값 | 조정 효과 |
| --- | ---: | --- |
| `num_steps_per_env` | 24 | rollout horizon |
| `max_iterations` | 8000 | 최대 PPO update 수 |
| `save_interval` | 50 | checkpoint 주기 |
| hidden dims | `[256, 128, 64]` | actor/critic MLP 크기 |
| activation | `elu` | MLP 활성함수 |
| `value_loss_coef` | 1.0 | critic loss 비중 |
| `clip_param` | 0.2 | PPO clipping 범위 |
| `entropy_coef` | 0.006 | 탐색을 위한 entropy 보너스 |
| learning epochs / mini-batches | 5 / 4 | rollout당 optimization 반복 |
| `learning_rate` | `1e-3` | PPO optimizer learning rate |
| `gamma` | 0.98 | reward discount factor |
| `lam` | 0.95 | GAE lambda |
| `desired_kl` | 0.01 | adaptive schedule의 KL 목표 |
| `max_grad_norm` | 1.0 | gradient clipping |

등록 task ID는 `Isaac-High-Level-Policy-Direct-Test-v0`입니다. Isaac Lab의 RSL-RL runner 경로에서 다음 형태로 실행합니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-High-Level-Policy-Direct-Test-v0 \
  --num_envs 64 \
  --headless
```

24 GB VRAM에서 논문 통합 환경 재현값은 64 vectorized environments입니다. 현재 config의 `num_envs=4096`는 640x480 tiled RGB + FCN을 함께 사용할 때 메모리 초과 가능성이 매우 높으므로 CLI 또는 config에서 줄여 시작하십시오.

> Gym 등록의 `rsl_rl_cfg_entry_point`는 현재 `rsl_rl_ppo_cfg` 모듈을 가리키지만 실제 파일명은 `rsl_rl_cfg.py`입니다. 학습 전 entry point를 실제 모듈명과 일치시키거나 파일명을 맞춰야 합니다.

---

## Legacy branch and Direct/Manager notes

- In IROL_SKY branch
    - Add stage
    - Commit with your message
    - Push to remote branch

```bash
git add .
git commit -m "<YOUR-COMMENT>"
git push origin IROL_SKY
```

- If your current branch is not IROL_SKY
```bash
git branch -a
(FIND YOUR BRANCH)
git checkout <YOUR-BRANCH>
```

- Update remote branch & update master branch (sync to remote)
```bash
git remote update
git checkout master
git pull origin master
```

```bash
git merge <YOUR-BRANCH>
```

> Conflict Fixing

- Push to master branch
```bash
git add .
git commit -m "<YOUR-COMMENT>"
git push origin master
```

<hr />

# Differences between <i>Manager Based RL</i> and <i>Direct RL</i>


## 1. DirectRLEnv

<p>
The custom class which is inherited DirectRLEnv have to defince some abstract method in order to build RL structure.
</p>

<ul>
    <li>_setup_scene</li>
    <li>_pre_physics_step</li>
    <li>_apply_action</li>
    <li>_get_observations</li>
    <li>_get_states</li>
    <li>_get_rewards</li>
    <li>_get_dones</li>
    <li>_set_debug_vis_impl</li>
</ul>


```python
import gymnasium as gym
from .common import VecEnvObs, VecEnvStepReturn

class DirectRLEnv(gym.Env):
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        (...)
        """
        return self._get_observations(), self.extras

    def step(self, action: torch.Tensor):
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


    def render(self, recompute: bool = False) -> np.ndarray | None:
        """
        (...)
        """
        return None

    def close(self):
        """Cleanup for the environment."""
        if not self._is_closed:
            # close entities related to the environment
            # note: this is order-sensitive to avoid any dangling references
            if self.cfg.events:
                del self.event_manager
            del self.scene
            if self.viewport_camera_controller is not None:
                del self.viewport_camera_controller
            # clear callbacks and instance
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            # destroy the window
            if self._window is not None:
                self._window = None
            # update closing status
            self._is_closed = True
```


## 2. ManagerBasedRLEnv

<p>
The custum class which is inherited ManagerBasedEnv uses methods that are defined in <b>Manager class</b>
</p>
<p>
In details, in manager-based method, parent class ManagerBasedEnv loads all managers which are defined in cfg, which has type called ManagerBasedEnvCfg.
</p>


```python
class ManagerBasedEnv:
    def __init__(self, cfg: ManagerBasedEnvCfg):
        """
        (...)
        """

    def load_managers(self):
        # prepare the managers
        # -- recorder manager
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)
        # -- event manager
        self.event_manager = EventManager(self.cfg.events, self)
        print("[INFO] Event Manager: ", self.event_manager)

        # perform events at the start of the simulation
        # in-case a child implementation creates other managers, the randomization should happen
        # when all the other managers are created
        if self.__class__ == ManagerBasedEnv and "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
        }



from .manager_based_env_cfg import ManagerBasedEnvCfg

class ManagerBasedRLEnv(ManagerBasedEnv, gym.Env):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)

        """
        (...)
        """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """
        (...)
        """
        return None
```


## 3. Conclustion

<p>
Both DirectRLEnv and ManagerBasedRLEnv are inherited gym.Env in order to build RL structure. However, the DirectRLEnv implements <i>action-observation-reward</i> structure via abstract class to control directly. Moreover, ManagerBasedRLEnv implements such methods via <i>manager</i>. Each manager is inherited the abstract class <b>ManagerBase</b> to implement nessecery fuctions.
</p>

---

## DRL_experiment README

The following section mirrors `src/DRL_experiment/README.md`.

# DRL-Occluded-Object-Search

ROS 2 기반의 **가려진 객체 탐색 및 로봇 조작 파이프라인**입니다. 이 저장소는 카메라 인지, Depth/PointCloud 기반 환경 상태 추정, FCN/DRL 기반 의사결정, Drop Grid, UR5e + Robotiq 제어 구성을 함께 다룹니다.

상세 설명은 최상단 README에 모두 넣지 않고 `docs/` 문서로 분리했습니다. README는 프로젝트를 빠르게 파악하고 필요한 문서로 이동하는 허브 역할을 합니다.

## 문서 바로가기

- [프로젝트 전체 설명](src/DRL_experiment/docs/project-overview.md)
- [디렉토리 구조 설명](src/DRL_experiment/docs/directory-structure.md)
- [아키텍처 및 노드 구동 원리](src/DRL_experiment/docs/architecture.md)
- [카메라 관련 메모](src/DRL_experiment/docs/camera.md)
- [명령어 메모](src/DRL_experiment/docs/commands.md)
- [이전 README 참고 문서](src/DRL_experiment/docs/last-readme.md)

## 한눈에 보는 구성

```text
RGB / Depth / PointCloud
        |
        v
object_tracker  ──>  fcn_network  ──>  robot_control
        |                  |
        |                  ├── FCN prediction
        |                  ├── DRL policy action
        |                  └── Drop grid decision
        |
        └── YOLO segmentation / closest object classification
```

## 주요 패키지

| 패키지 | 역할 |
| --- | --- |
| `base_package` | 이미지, 객체, transform 등 공통 유틸리티 |
| `custom_msgs` | 프로젝트 전용 ROS msg/srv 정의 |
| `object_tracker` | YOLO segmentation, closest object, 통합 이미지 디버깅 |
| `fcn_network` | FCN, DRL, grid, drop grid 추론 노드 |
| `mcts` | MCTS 기반 의사결정 실험 |
| `robot_control` | UR5e + Robotiq 상위 제어 |
| `ros2_robotiq_gripper-humble` | Robotiq gripper ROS 2 Humble 연동 패키지 |
| `ur5e_robotiq_config` | UR5e + Robotiq MoveIt 설정 |

## 빠른 시작 방향

1. ROS 2 workspace로 저장소를 준비합니다.
2. `requirements.txt`와 각 ROS 패키지 의존성을 설치합니다.
3. `colcon build`로 workspace를 빌드합니다.
4. `docs/commands.md`와 각 launch 파일을 참고해 카메라, 인지, 추론, 제어 노드를 실행합니다.

자세한 노드별 topic/service 관계는 [아키텍처 및 노드 구동 원리](src/DRL_experiment/docs/architecture.md)를 참고하세요.

