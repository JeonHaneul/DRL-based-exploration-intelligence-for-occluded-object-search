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

External open-source dependencies such as Isaac Lab, ViSP, MoveIt messages, Web Video Server, Robotiq, and Serial remain Git submodules. Clone them with:

```bash
git submodule update --init --recursive
```

---

## DRL_training README

The following section mirrors `src/DRL_training/README.md`.

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

---

# Differences between *Manager Based RL* and *Direct RL*

## 1. DirectRLEnv

The custom class which is inherited DirectRLEnv have to defince some abstract method in order to build RL structure.

- _setup_scene
- _pre_physics_step
- _apply_action
- _get_observations
- _get_states
- _get_rewards
- _get_dones
- _set_debug_vis_impl

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
        # add action noise if self.cfg.action_noise_model:
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
            # If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1
        self.common_step_counter += 1
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

The custum class which is inherited ManagerBasedEnv uses methods that are defined in Manager class

In details, in manager-based method, parent class ManagerBasedEnv loads all managers which are defined in cfg, which has type called ManagerBasedEnvCfg.

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
            # If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1
        self.common_step_counter += 1
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

Both DirectRLEnv and ManagerBasedRLEnv are inherited gym.Env in order to build RL structure. However, the DirectRLEnv implements action-observation-reward structure via abstract class to control directly. Moreover, ManagerBasedRLEnv implements such methods via manager. Each manager is inherited the abstract class ManagerBase to implement nessecery fuctions.

---

## DRL_experiment README

The following section mirrors `src/DRL_experiment/README.md`.

# DRL-Occluded-Object-Search

ROS 2 기반의 **가려진 객체 탐색 및 로봇 조작 파이프라인**입니다. 이 저장소는 카메라 인지, Depth/PointCloud 기반 환경 상태 추정, FCN/DRL 기반 의사결정, Drop Grid, UR5e + Robotiq 제어 구성을 함께 다룹니다.

상세 설명은 최상단 README에 모두 넣지 않고 `docs/` 문서로 분리했습니다. README는 프로젝트를 빠르게 파악하고 필요한 문서로 이동하는 허브 역할을 합니다.

## 문서 바로가기

- [프로젝트 전체 설명](docs/project-overview.md)
- [디렉토리 구조 설명](docs/directory-structure.md)
- [아키텍처 및 노드 구동 원리](docs/architecture.md)
- [카메라 관련 메모](docs/camera.md)
- [명령어 메모](docs/commands.md)
- [이전 README 참고 문서](docs/last-readme.md)

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

자세한 노드별 topic/service 관계는 [아키텍처 및 노드 구동 원리](docs/architecture.md)를 참고하세요.

