import argparse

from mlagents_envs.environment import UnityEnvironment

def unity_arg_parser():
    """
    Create an argparse.ArgumentParser for configuring Unity environment.
    """
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "--env",
        default=None,
        dest="env_path",
        help="Path to the Unity executable to train",
    )
    argparser.add_argument(
        "--seed",
        default=-1,
        type=int,
        help="A number to use as a seed for the random number generator used by the training code",
    )
    argparser.add_argument(
        "--base-port",
        default=UnityEnvironment.BASE_ENVIRONMENT_PORT,
        type=int,
        help="The starting port for Unity environment communication.",
    )


    eng_conf = argparser.add_argument_group(title="Unity Engine Configuration")
    eng_conf.add_argument(
        "--width",
        default=None,
        type=int,
        help="The width of the executable window of the environment(s) in pixels "
        "(ignored for editor training).",
    )
    eng_conf.add_argument(
        "--height",
        default=None,
        type=int,
        help="The height of the executable window of the environment(s) in pixels "
        "(ignored for editor training)",
    )
    eng_conf.add_argument(
        "--quality-level",
        default=5,
        type=int,
        help="The quality level of the environment(s). Equivalent to calling "
        "QualitySettings.SetQualityLevel in Unity.",
    )
    eng_conf.add_argument(
        "--time-scale",
        default=20,
        type=float,
        help="The time scale of the Unity environment(s). Equivalent to setting "
        "Time.timeScale in Unity.",
    )
    eng_conf.add_argument(
        "--target-frame-rate",
        default=-1,
        type=int,
        help="The target frame rate of the Unity environment(s). Equivalent to setting "
        "Application.targetFrameRate in Unity.",
    )
    eng_conf.add_argument(
        "--capture-frame-rate",
        default=60,
        type=int,
        help="The capture frame rate of the Unity environment(s). Equivalent to setting "
        "Time.captureFramerate in Unity.",
    )
    return argparser


def chickAI_arg_parser():
    """
    Create an argparse.ArgumentParser for ChickAI environment.
    """
    argparser = unity_arg_parser()
    env_conf = argparser.add_argument_group(title="ChickAI Environment Configuration")
    env_conf.add_argument(
        "--input-resolution",
        default=64,
        type=int,
        help="Resolution of visual inputs to agent."
    )
    env_conf.add_argument(
        "--episode-steps",
        default=1000,
        type=int,
        help="Number of steps per episode."
    )
    env_conf.add_argument(
        "--video1",
        default=None,
        type=str,
        help="Path to the first video file to play."
    )
    env_conf.add_argument(
        "--video2",
        default=None,
        type=str,
        help="Path to the second video file to play."
    )
    env_conf.add_argument(
        "--log-dir",
        default=None,
        type=str,
        help="Directory to save environment logs"
    )
    env_conf.add_argument(
        "--test-mode",
        action='store_true',
        help="Run ChickAI environment in Test Mode."
    )
    return argparser


if __name__=="__main__":
    options = chickAI_arg_parser().parse_args()
