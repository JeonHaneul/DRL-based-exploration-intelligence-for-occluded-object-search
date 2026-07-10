# DRL Training

## 학습 데이터 생성

학습용 데이터셋 생성 코드는 `src/high_level_dataset`에 있습니다. Isaac Lab 기반 시뮬레이션에서 선반 장면과 타깃 객체의 배치 후보를 생성한 뒤, 이를 결합해 최종 학습용 distribution map을 만듭니다.

### 생성 코드

| 파일 | 역할 |
| --- | --- |
| `Scene_generator.py` | 다양한 객체가 배치된 선반 장면을 생성하고 RGB, depth, segmentation 및 scene mask 데이터를 저장합니다. |
| `Target_scene_gnerator.py` | 타깃 객체가 선반에 위치할 수 있는 영역과 자세를 순회하며 ground-truth용 target scene 데이터를 생성합니다. |
| `Final_distribution_map_generator.py` | scene/target depth와 similarity mask를 처리하고 결합하여 최종 학습 데이터인 distribution map을 생성합니다. |

### 생성 순서

1. `Scene_generator.py`로 학습용 선반 장면과 mask를 생성합니다.
2. `Target_scene_gnerator.py`로 타깃 객체의 배치 가능 영역에 대한 ground truth를 생성합니다.
3. `Final_distribution_map_generator.py`로 두 결과를 결합해 최종 distribution map을 생성합니다.

### 실행 예시

```bash
./isaaclab.sh -p source/standalone/shelf_env/Scene_generator.py --target_object can_2 --enable_camera --save --num_img 100
./isaaclab.sh -p source/standalone/shelf_env/Target_scene_gnerator.py --target_object can_3 --enable_camera --save --row 4
./isaaclab.sh -p source/standalone/shelf_env/Final_distribution_map_generator.py --target_object can_2 --save
```

> 실행 전 각 스크립트의 Isaac Lab 경로, Omniverse USD asset 경로, 출력 폴더를 현재 환경에 맞게 설정해야 합니다.

---

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