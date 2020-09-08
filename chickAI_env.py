import os
from typing import List, Optional

import numpy as np

import gym

from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException, UnityWorkerInUseException
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfig,
    EngineConfigurationChannel,
)

class ChickAIGymEnv(gym.Wrapper):
    def __init__(self, env, agent_info_channel):
        super().__init__(env)
        self.env = env
        self.agent_info_channel = agent_info_channel

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        agent_info = self.agent_info_channel.get_property_dict_copy()
        info.update(dict(agent=agent_info))
        return next_state, reward, done, info


class PytorchVisualEnv(gym.ObservationWrapper):
    """
    Gym env wrapper that transpose visual observations to (C,H,W).
    """
    def __init__(self, env):
        super().__init__(env)

        # Assumes visual observation space with shape (H, W, C)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert len(env.observation_space.shape) == 3

        h, w, c = env.observation_space.shape
        if env.observation_space.dtype == np.uint8:
            self.observation_space = gym.spaces.Box(0, 255, dtype=np.uint8, shape=(c, h, w))
        else:
            self.observation_space = gym.spaces.Box(0, 1, dtype=np.float32, shape=(c, h, w))

    def observation(self, obs):
        return obs.transpose((2, 0, 1))


def make_chickAI_gym_env(options):
    """
    Build ChickAI Gym Environment from command line options.
    """
    unity_env, agent_info_channel = make_chickAI_unity_env(options)
    env = UnityToGymWrapper(unity_env, flatten_branched=True, use_visual=True, uint8_visual=True)
    env = ChickAIGymEnv(env, agent_info_channel)
    return env


def make_chickAI_unity_env(options):
    """
    Build ChickAI UnityEnvironment from command line options.
    """
    engine_config = EngineConfig(
        width=options.width,
        height=options.height,
        quality_level=options.quality_level,
        time_scale=options.time_scale,
        target_frame_rate=options.target_frame_rate,
        capture_frame_rate=options.capture_frame_rate,
    )
    env_args = _build_chickAI_env_args(
        input_resolution=options.input_resolution,
        episode_steps=options.episode_steps,
        video_1_path=options.video1,
        video_2_path=options.video2,
        log_dir=options.log_dir,
        test_mode=options.test_mode
    )
    # Set up FloatPropertiesChannel to receive auxiliary agent information.
    agent_info_channel = FloatPropertiesChannel()
    unity_env = make_unity_env(
        env_path=options.env_path,
        port=options.base_port,
        seed=options.seed,
        env_args=env_args,
        engine_config=engine_config,
        side_channels=[agent_info_channel]
    )
    return unity_env, agent_info_channel


def make_unity_env(
    env_path: Optional[str] = None,
    port: int = UnityEnvironment.BASE_ENVIRONMENT_PORT,
    seed: int = -1,
    env_args: Optional[List[str]] = None,
    engine_config: Optional[EngineConfig] = None,
    side_channels: Optional[List[SideChannel]]= None
) -> UnityEnvironment:
    """
    Create a UnityEnvironment.
    """
    # Use Unity Editor if env file is not provided.
    if env_path is None:
        port = UnityEnvironment.DEFAULT_EDITOR_PORT
    else:
        launch_string = UnityEnvironment.validate_environment_path(env_path)
        if launch_string is None:
            raise UnityEnvironmentException(
                f"Couldn't launch the {env_path} environment. Provided filename does not match any environments."
            )

    # Configure Unity Engine.
    if engine_config is None:
        engine_config = EngineConfig.default_config()

    engine_configuration_channel = EngineConfigurationChannel()
    engine_configuration_channel.set_configuration(engine_config)

    if side_channels is None:
        side_channels = [engine_configuration_channel]
    else:
        side_channels.append(engine_configuration_channel)

    while True:
        try:
            env = UnityEnvironment(
                file_name=env_path,
                seed=seed,
                base_port=port,
                args=env_args,
                side_channels=side_channels,
            )
            break
        except UnityWorkerInUseException:
            port += 1
        else:
            break

    return env


def _build_chickAI_env_args(
    input_resolution: Optional[int] = None,
    episode_steps: Optional[int] = None,
    video_1_path: Optional[str] = None,
    video_2_path: Optional[str] = None,
    log_dir: Optional[str] = None,
    test_mode: bool = False,
) -> List[str]:
    """
    Build environment arguments that will be passed to chickAI unity environment.
    """
    # Always enable agent info side channel.
    env_args = ["--enable-agent-info-channel"]

    if input_resolution is not None:
        env_args.append("--input-resolution")
        env_args.append(str(input_resolution))

    if episode_steps is not None:
        env_args.append("--episode-steps")
        env_args.append(str(episode_steps))

    if video_1_path is not None:
        abs_path = os.path.abspath(os.path.expanduser(video_1_path))
        video_1_url = f"file://{abs_path}"
        env_args.append("--video1")
        env_args.append(video_1_url)

    if video_2_path is not None:
        abs_path = os.path.abspath(os.path.expanduser(video_2_path))
        video_2_url = f"file://{abs_path}"
        env_args.append("--video2")
        env_args.append(video_2_url)

    if log_dir is not None:
        env_args.append("--log-dir")
        env_args.append(log_dir)

    if test_mode is True:
        env_args.append("--test-mode")

    return env_args
