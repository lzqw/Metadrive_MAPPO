"""
# @Time    : 2021/7/1 8:44 上午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_wrappers.py
Modified from OpenAI Baselines code to work with multi-agent envs
"""

import numpy as np


# single env
class DummyVecEnv():
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.actions = None

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        # print(dones)
        rews = np.expand_dims(rews, axis=2)


        for i, (env_done, env_info) in enumerate(zip(dones, infos)):
            if isinstance(env_done, bool):
                if env_done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(env_done):
                    obs[i] = self.envs[i].reset()   # reset the env
                # # Handle multi-agent environments
                # for agent_index, agent_done in enumerate(env_done):
                #     if agent_done:
                #         # Check if the agent has reached its destination
                #         if env_info[agent_index].get('arrive_dest'):
                #             # Replace the data of the first agent that reached its destination with the new agent's data
                #             pass
                #         else:
                #             # If the agent is done but hasn't reached its destination, reset the environment
                #             obs[i] = self.envs[i].reset()
                #             break  # Assuming the entire environment is reset

        self.actions = None
        # print("----->",dones)
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs]  # [env_num, agent_num, obs_dim]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError
