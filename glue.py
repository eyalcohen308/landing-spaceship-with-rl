#!/usr/bin/env python

"""Glues together an experiment, agent, and environment.
"""

from __future__ import print_function


class RLGlue:
    """RLGlue class

    args:
        env_name (string): the name of the module where the Environment class can be found
        agent_name (string): the name of the module where the Agent class can be found
    """

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.total_reward = None
        self.last_action = None
        self.num_steps = None
        self.num_episodes = None

    def init(self, seed):
        self.env.init()
        self.agent.init(seed)

        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def start(self):
        self.total_reward = 0.0
        self.num_steps = 1

        last_state = self.env.start()
        self.last_action = self.agent.start(last_state)

        observation = (last_state, self.last_action)

        return observation

    def step(self):
        (reward, last_state, term) = self.env.step(self.last_action)

        self.total_reward += reward

        if term:
            self.num_episodes += 1
            self.agent.end(reward)
            roat = (reward, last_state, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.step(reward, last_state)
            roat = (reward, last_state, self.last_action, term)

        return roat

    def episode(self, max_steps_this_episode):
        is_terminal = False

        self.start()

        while (not is_terminal) and ((max_steps_this_episode == 0) or
                                     (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.step()
            is_terminal = rl_step_result[3]

        return self.agent.sum_rewards
