import numpy as np
import matplotlib.pyplot as plt
from .multiagentenv import MultiAgentEnv

class Environment(MultiAgentEnv):

    def __init__(self, env_name='matthew', seed=None):

        self._seed = seed
        self.normalize = True
        self.resource_type = 'all'
        self.obs3neighbors = True
        self.map_name = "1c3s5z"
        
        self.env_name = env_name
        if self.env_name == "matthew":
            from common.env_matthew import Env
            self.env = Env(self.normalize, self.resource_type, self.obs3neighbors)
        elif self.env_name == "sumo":
            from common.env_sumo import Env
            self.env = Env(self.normalize)
        elif self.env_name == "starcraft":
            from common.env_starcraft import Env
            self.env = Env(self.map_name)
        elif self.env_name == "job":
            from common.env_job import Env
            self.env = Env(self.normalize, self.resource_type)
        else:
            print('Invalid Env')
        
        self.episode_limit = self.env.max_steps
        self.n_agents = self.env.n_agent
        self.n_actions = self.env.n_actions

        self._episode_steps = 0
        self.run = 0
        self.last_action = [np.zeros(self.n_actions) for _ in range(self.n_agents)]
        
        self.env.reset()
        super(Environment, self).__init__()

    def step(self, actions):
        """ Returns reward, terminated, info """
        self._episode_steps += 1
        actions = [int(a) for a in actions]
        obs, reward, done = self.env.step(actions)
        reward = np.sum(reward)
        terminated = done or self._episode_steps >= self.episode_limit
        info = {}

        for agent_id, action in enumerate(actions):
            self.last_action[agent_id] = np.zeros(self.n_actions)
            self.last_action[agent_id][action] = 1.

        return reward, terminated, info
    
    def get_obs(self):
        obs_n = self.env._get_obs()
        return obs_n
    
    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.get_obs_agent(0))
    
    def get_state(self):
        return np.concatenate(self.get_obs())
    
    def get_state_size(self):
        """ Returns the shape of the state"""
        return (self.get_obs_size() * self.n_agents)

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]
    
    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)
    
    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def reset(self):
        self._episode_steps = 0
        if self.run == 0:
            print('Game Start')
            self.env.close()
        else:
            self.env.end_episode()

        self.run += 1
        self.env.reset()
        self.last_action = [np.zeros(self.n_actions) for _ in range(self.n_agents)]
        return self.get_obs(), self.get_state()

    def get_stats(self): 
        return None

    def render(self):
        for i in range(self.n_agents):
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = self.ant[i][0] + self.size[i] * np.cos(theta)
            y = self.ant[i][1] + self.size[i] * np.sin(theta)
            plt.plot(x, y)
        for i in range(self.n_resource):
            plt.scatter(self.resource[i][0], self.resource[i][1], color='green')
        plt.axis("off")
        plt.axis("equal")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.ion()
        plt.pause(0.1)
        plt.close()

    def close(self):
        self.env.close()

    def seed(self):
        return self._seed

    def save_replay(self):
        pass
    
    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
