# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
from os import path
#print(currentdir)
parentdir =path.abspath(path.join(__file__ ,"../../../../../.."))
os.sys.path.insert(0, parentdir)
print(parentdir)

from envs.panda_envs.panda_reach_gym_env import pandaReachGymEnv
from stable_baselines import logger
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
from termcolor import colored

import datetime
import pybullet_data
import robot_data
import numpy as np
import time
import math as m
import gym
import sys, getopt

def main(argv):
    fixed = True

    policy_name = "sac_reaching_policy"

    obj_pose_rnd_std = 0 if fixed == True else 0.05
    pandaenv = pandaReachGymEnv(renders=True, use_IK=0, numControlledJoints=7, obj_pose_rnd_std=obj_pose_rnd_std, includeVelObs=True)
    n_actions = pandaenv.action_space.shape[-1]

    pandaenv = DummyVecEnv([lambda: pandaenv])

    model = SAC(MlpPolicy, pandaenv, gamma=0.9, batch_size=16, verbose=1, tensorboard_log="../pybullet_logs/pandareach_sac/")

    model.learn(total_timesteps=1000000)

    model.save("../pybullet_logs/pandareach_sac/"+ policy_name)

    del model # remove to demonstrate saving and loading

if __name__ == '__main__':
    main(sys.argv[1:])

