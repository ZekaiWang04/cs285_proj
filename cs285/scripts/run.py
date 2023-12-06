import os
import time
from scripting_utils import make_logger, make_config
import argparse
from cs285.scripts.run_mpc import run_training_loop_mpc
from cs285.scripts.run_ode import run_training_loop_ode

from cs285.envs import register_envs

register_envs()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)

    args = parser.parse_args()

    config, agent_name = make_config(args.config_file)
    logger = make_logger(config)

    if agent_name == "mpc":
        run_training_loop_mpc(config, agent_name, logger, args)
    elif agent_name == "ode":
        run_training_loop_ode(config, agent_name, logger, args)
    else:
        raise Exception

if __name__ == "__main__":
    main()
