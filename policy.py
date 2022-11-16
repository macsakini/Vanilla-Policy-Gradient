from collections import deque

import numpy as np
import torch
import torch.distributions as td
import random
import matplotlib.pyplot as plt

from model import VPGModel
from paddle import PingPong

# Q-Learning
# q_value = rewards + self.gamma * q_val
env = PingPong()
np.random.seed(0)


class VPGPolicy():
    def __init__(self, state_space, action_space, lr):
        self.lr = lr
        self.epsilon = 1
        self. epsilon_decrement = 0.01
        self.batch_size = 5
        self.gamma = .95
        self.state_space = state_space
        self.action_space = action_space
        self.model = VPGModel(state_space=self.state_space,
                              action_space=self.action_space, lr=self.lr)
        self.memory = deque(maxlen=100000)

    def act(self, state):
        if self.epsilon > random.random():
            act = random.randint(0, 2)
            return act
        logits = self.model(state)
        act = td.Categorical(logits).sample().item()
        return act

    def trajectory(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def reward_to_go(self, rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Explore trajectory
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([i[0] for i in minibatch]).astype(np.float32)
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch]).astype(np.float32)
        dones = np.array([i[4] for i in minibatch])

        # memory
        states_t = torch.tensor(np.squeeze(states))
        actions_t = torch.tensor(np.squeeze(actions))
        rewards_t = torch.tensor(np.squeeze(rewards))
        next_states_t = torch.tensor(np.squeeze(next_states))
        dones_t = torch.tensor(np.squeeze(dones))

        self.model.optimizer.zero_grad()

        rewards_to_go = torch.tensor(list(self.reward_to_go(rewards)))

        actions_t = actions_t.reshape([5, 1])

        # Advantage
        # Q(s,a) - V(s)
        pi_s = self.model(states_t)

        pi_s_a = self.model(next_states_t)

        q_s_a = np.sum(self.gamma * rewards_to_go.numpy())

        v_s = np.sum(self.gamma * rewards_to_go.numpy())

        advantage = q_s_a - v_s

        print(q_s_a)

        self.model.optimizer.step()

        self.epsilon -= self.epsilon_decrement


def train_dqn(episode):

    loss = []

    action_space = 3
    state_space = 5
    max_steps = 1000

    agent = VPGPolicy(state_space, action_space, 1e-3)
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, state_space))
        state = torch.from_numpy(state).to(torch.float32)
        score = 0
        for i in range(max_steps):
            action = agent.act(state)
            reward, next_state, done = env.step(action)
            score += reward
            next_state = np.reshape(
                next_state, (1, state_space)).astype(np.float32)
            state = state.tolist()
            agent.trajectory(state, action, reward, next_state, done)
            state = torch.tensor(next_state)
            agent.learn()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)
    return loss


if __name__ == '__main__':

    ep = 500
    loss = train_dqn(ep)
    plt.plot([i for i in range(ep)], loss)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
