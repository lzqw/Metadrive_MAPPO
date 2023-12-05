import random
import time

import numpy as np
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv,MultiAgentStraightEnv
)
import argparse
from metadrive.constants import HELP_MESSAGE
from metadrive.policy.idm_policy import ManualControllableIDMPolicy

envs = dict(
    roundabout=MultiAgentRoundaboutEnv,
    intersection=MultiAgentIntersectionEnv,
    tollgate=MultiAgentTollgateEnv,
    bottleneck=MultiAgentBottleneckEnv,
    parkinglot=MultiAgentParkingLotEnv,
    pgma=MultiAgentMetaDrive,
    straight=MultiAgentStraightEnv
)

class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self,args,config):
        self.args=args
        env_cls_name = args.env
        self.env= envs[env_cls_name](
            config
         )
        self.agent_num = self.args.num_agents
        self.obs_dim = list(self.env.observation_space.values())[0].shape[0]
        self.action_dim = list(self.env.action_space.values())[0].shape[0]
        self.action_space = list(self.env.action_space.values())
        self.observation_space=list(self.env.observation_space.values())
        self.need_reset=False



    def reset(self):

        state = self.env.reset()
        sub_agent_obs=list(state[0].values())

        return sub_agent_obs

    def step(self, actions):
        sub_agent_obs,reward,done,truncateds,info = self.env.step({agent_id: action for agent_id,action in zip(self.env.vehicles.keys(),actions)})
        sub_agent_obs=list(sub_agent_obs.values())
        sub_agent_reward=list(reward.values())
        sub_agent_done = list(done.values())[:-1]
        sub_agent_info=list(info.values())
        # print(sub_agent_done)

        new_agent_processed = False  # Flag to track if new agent's data is already used
        for agent_index, agent_done in enumerate(sub_agent_done):
            if agent_done:
                if len(sub_agent_done)>self.agent_num :
                    # if len(sub_agent_done)!=self.agent_num:
                    sub_agent_obs[agent_index] = sub_agent_obs[-1]
                    sub_agent_obs = np.delete(sub_agent_obs, -1, axis=0)
                    sub_agent_reward = np.delete(sub_agent_reward, -1)
                    sub_agent_done = np.delete(sub_agent_done, -1)
                    sub_agent_info = np.delete(sub_agent_info, -1)
                    new_agent_processed = True  # Mark that new agent's data has been used
                elif new_agent_processed and len(sub_agent_done)==self.agent_num:
                    # If another agent is done and the new agent's data is already used, skip processing
                    continue
                else:
                    # If the agent is done but hasn't reached its destination, reset the environment
                    # ref = my_env.reset()
                    break  # Assuming the entire environment is reset
        if len(sub_agent_done)<self.agent_num:
            print("len(sub_agent_done)<self.agent_num",len(sub_agent_done),len(sub_agent_obs))
        # print(reward)
        while len(sub_agent_done)<self.agent_num:
            sub_agent_done=np.append(sub_agent_done,False)  # If the agent is done but hasn't reached its destination, reset the environment
            sub_agent_reward=np.append(sub_agent_reward,0)
            # sub_agent_obs=np.append(sub_agent_obs,sub_agent_obs[-1])
            sub_agent_obs.append(sub_agent_obs[-1])
            sub_agent_info=np.append(sub_agent_info,sub_agent_info[-1])
            self.need_reset=True
            print("processing:",len(sub_agent_done),len(sub_agent_obs))
            print("Warning: Agent is done but hasn't reached its destination, reset the environment")


        # print(sub_agent_done, len(sub_agent_obs), len(sub_agent_reward), len(sub_agent_info))
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def close(self):
        self.env.close()


if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="intersection", choices=list(envs.keys()))
    parser.add_argument("--top_down", action="store_true",default=True)
    parser.add_argument("--num_agents", type=int,default=5)
    args = parser.parse_args()
    config=dict(
        horizon=200,
        use_render=True,
        crash_done= True,
        agent_policy=ManualControllableIDMPolicy,
        num_agents=args.num_agents,
        delay_done=False,
        vehicle_config=dict(
            lidar=dict(
                add_others_navi=False,
                num_others=4,
                distance=50,
                num_lasers=30,
            ),
            side_detector=dict(num_lasers=30),
            lane_line_detector=dict(num_lasers=12),

        )
    )
    my_env=EnvCore(args,config)
    my_env.reset()
    action=[np.zeros(2)+[0,1] for i in range(args.num_agents)]
    my_env.env.switch_to_third_person_view()  # Default is in Top-down vwwwwwwwwiew, we switch to Third-person view.
    while True:
        ref,r,done,info=my_env.step(action)
        print(len(done),len(ref))
        # print(my_env.env.episode_step,my_env.env.config["horizon"])
        if np.all(done) or my_env.env.episode_step >= my_env.env.config["horizon"]:
            ref = my_env.env.reset()  # reset the en

