import jax
import jax.numpy as jp
import mujoco
import mujoco.mjx as mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import humanoid, common
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import register_environment
from ml_collections import config_dict


def default_config() -> config_dict.ConfigDict:
    return humanoid.default_config()


class HumanoidBabyFreeze(humanoid.Humanoid):
    """Humanoid environment for a static baby freeze pose."""

    def __init__(self, config: config_dict.ConfigDict = default_config(), config_overrides=None):
        # Zero move speed so we reuse parent's stand reward utilities
        super().__init__(move_speed=0.0, config=config, config_overrides=config_overrides)
        # target pose: lying horizontally on the ground
        quat = jp.array([jp.sqrt(0.5), 0.0, -jp.sqrt(0.5), 0.0])
        pos = jp.array([0.0, 0.0, 0.3])
        self._target_qpos = jp.concatenate([pos, quat, jp.zeros(self.mjx_model.nq - 7)])

    def reset(self, rng: jax.Array) -> mjx_env.State:
        data = mjx_env.init(self.mjx_model)
        data = data.replace(qpos=self._target_qpos)
        metrics = {
            "reward/pose": jp.zeros(()),
            "reward/small_control": jp.zeros(()),
        }
        info = {"rng": rng}
        reward_value, done = jp.zeros(2)
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward_value, done, metrics, info)

    def _get_reward(self, data: mjx.Data, action: jax.Array, info: dict[str, any], metrics: dict[str, any]) -> jax.Array:
        del info
        pose_error = jp.mean(jp.square(data.qpos - self._target_qpos))
        pose_reward = reward.tolerance(pose_error, bounds=(0.0, 0.01), margin=1.0, sigmoid="quadratic")
        metrics["reward/pose"] = pose_reward

        small_control = reward.tolerance(action, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        metrics["reward/small_control"] = small_control

        return pose_reward * small_control


# Register the environment so it can be loaded via mujoco_playground.registry
register_environment("HumanoidBabyFreeze", HumanoidBabyFreeze, default_config)
