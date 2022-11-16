from model import VPGModel

import torch
import random
import numpy as np
import torch.distributions as td
from collections import deque
import matplotlib.pyplot as plt

from paddle import PingPong


class VPGPolicy():
    def __init__(self, state_space, action_space):
        self.batch_size = 64
        self.gamma = .95
        self.memory = []  # deque(maxlen=100000)
        self.model = VPGModel(
            state_space=state_space, action_space=action_space, lr=1e-3)

    def act(self, state):
        probs = self.model(state)
        catgor_sample = td.Categorical(probs)
        act = catgor_sample.sample().item()
        return act

    def remember(self, states, actions, weights, returns, lengths):
        self.memory.append([states, actions, weights, returns, lengths])

    def compute_loss(self, states, actions, weights):
        logp = self.model(states)
        logp = td.Categorical(logp)
        y_pred = -(logp * weights)
        return y_pred.mean()

    def learn(self):
        # if len(self.memory) > 2000:
        #     self.memory.pop(0)

        minibatch = random.sample(self.memory, min(
            len(self.memory), self.batch_size))

        states_np = np.array([i[0] for i in minibatch])
        actions_np = np.array([i[1] for i in minibatch])
        rewards_np = np.array([i[2] for i in minibatch])
        next_states_np = np.array([i[3] for i in minibatch])
        dones_np = np.array([i[4] for i in minibatch])

        states_np = np.squeeze(states_np)
        actions_np = np.squeeze(actions_np)

        states = torch.from_numpy(np.squeeze(states_np)).to(torch.float32)
        actions = torch.from_numpy(np.squeeze(actions_np))
        rewards = torch.from_numpy(np.squeeze(rewards_np))
        dones = torch.from_numpy(np.squeeze(dones_np))
        next_states = torch.from_numpy(
            np.squeeze(next_states_np)).to(torch.float32)

        R = torch.tensor([np.sum(rewards_np[i:]*(self.gamma**np.array(range(i, len(rewards_np)))))
                         for i in range(len(rewards_np))])

        probs = self.model(states)

        sampler = td.Categorical(probs)
        # "-" because it was built to work with gradient descent, but we are using gradient ascent
        log_probs = sampler.log_prob(actions)
        # loss that when differentiated with autograd gives the gradient of J(θ)
        pseudo_loss = torch.sum(log_probs * R)
        # update policy weights
        self.model.optimizer.zero_grad()
        pseudo_loss.backward()
        self.model.optimizer.step()


game = PingPong()


def train_dqn(episode):

    loss = []

    action_space = 3
    state_space = 5
    max_steps = 1000

    agent = VPGPolicy(state_space, action_space)
    for e in range(episode):
        state = game.reset()
        state = np.reshape(state, (1, state_space))
        state = torch.from_numpy(state).to(torch.float32)
        score = 0
        for i in range(max_steps):
            action = agent.act(state)
            reward, next_state, done = game.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, state_space))
            agent.remember(state.numpy(), action,
                           reward, next_state, done)
            state = next_state
            state = torch.from_numpy(next_state).to(torch.float32)
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
