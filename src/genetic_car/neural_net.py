import logging

import random
from collections import deque
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from genetic_car.plots import replay_run

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Force CUDA initialization by performing a dummy operation
_ = torch.randn(1).cuda()


# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        raw_output = self.fc(x)
        return raw_output


# Reinforcement Learning with DQN
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=10000)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state) -> np.ndarray:
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            # return actions as numpy array
            return self.policy_net(state).cpu().numpy()
            # return self.policy_net(state)

    def sample_prioritized_memory(self, batch_size, alpha=0.6):
        # give the priority to the samples with higher reward
        rewards = np.array([m[2] for m in self.memory])
        probs = np.abs(rewards) + 1e-5
        probs = probs ** alpha
        probs /= probs.sum()
        batch = random.choices(self.memory, k=batch_size, weights=probs)

        return batch

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # batch = random.sample(self.memory, batch_size)
        batch = self.sample_prioritized_memory(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states)
        next_q_values = self.target_net(next_states)
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# Environment Wrapper
class CarEnvironment:
    def __init__(self, car, ):
        self.car = car
        self.action_dim = 1

    def reset(self):
        self.car.reset()
        return self.get_state()

    def get_state(self) -> np.ndarray:
        distances = []
        for p in self.car.crash_points:
            diff_x = p.x - self.car.x
            diff_y = p.y - self.car.y
            distances.append(np.linalg.norm(np.asarray([diff_x, diff_y])))
        inputs = np.array(distances + [self.car.speed, self.car.direction], dtype=np.float32)
        return inputs

    def step(self, action, step_num: int) -> Tuple[np.ndarray, float, bool, dict]:
        direction_offset = action[0].item() * car.max_turn_angle

        prev_progress = car.progress_on_track()
        self.car.turn(direction_offset)
        self.car.move()
        next_progress = car.progress_on_track()

        done = False
        if next_progress < prev_progress:
            print("Going backwards")
            reward = -100
        elif self.car.is_crashed():
            print("Crashed")
            reward = 0
            done = True
        elif prev_progress < 1 and next_progress == 1:
            print("Crossed the finish line the right way")
            reward = 10000
            done = True
        else:
            distance_from_start = calc_distance(self.car.position, self.car.starting_point)
            reward = distance_from_start
        info = {
            "progress": next_progress,
            "direction_offset": direction_offset,
            "prev_progress": prev_progress,
            "reward": reward
        }
        return self.get_state(), reward, done, info


# Training Loop
def train_dqn(car:Car,  episodes=1000, max_steps=1500, batch_size=64):
    log.info("Training DQN")
    env = CarEnvironment(car)
    state_dim = len(env.get_state())
    agent = DQNAgent(state_dim, env.action_dim)
    best_reward = -np.inf
    best_progress = 0

    def record_run(car, run, **kwargs):
        car_dict = car.to_dict()
        car_dict.update(kwargs)
        run.append(car_dict)
        return run

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        run = []
        run = record_run(car, run, reward=0, action=0, progress=0)
        print(f"---------------------------------------------------- Episode {episode + 1}/{episodes}")
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action, step)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            run = record_run(car, run,
                             reward=reward,
                             action=action,
                             direction_offset=info['direction_offset'],
                             progress=info['progress'])
            # print(
            #     f"Step: {step:.2f} Reward: {reward:.2f}, total: {total_reward:.2f} prog: {info['progress']:.2f} prev: {info['prev_progress']:.2f}")

            if done:
                break
        print(f"total award: {total_reward:.2f} num steps: {step} progress: {info['progress']:.2f}")

        if total_reward > best_reward:
            best_reward = total_reward
            best_run = run
            print("======= New Best Reward =======")
            print(f"        Episode {episode + 1}/{episodes}, Reward: {total_reward}, steps: {step}")
            print("================================")
            if info['progress'] > best_progress + 0.1:
                replay_run(car, run)
            best_progress = max(best_progress, info['progress'])
        print("")
        agent.replay(batch_size)
        agent.update_target_net()
        if episode % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")
            replay_run(car, run)

    replay_run(car, best_run)
    return agent


if __name__ == "__main__":
    from genetic_car.track import Track
    from genetic_car.helpers import Point, calc_distance
    from genetic_car.car import Car
    import cv2
    import time
    from pathlib import Path

    # Example usage
    image = cv2.imread(str(Path(__file__).parent / "tracks" / "race_track_1.png"), cv2.IMREAD_GRAYSCALE)
    track = Track(image)
    start_point = track.center_path[track.get_closest_point_index((0, 0))]
    track.set_start_point(start_point)

    st = time.time()
    view_angles = list(np.arange(-90, 91, 30))

    start_direction = track.direction_at_point(start_point)
    car = Car(track=track,
              starting_point=Point(*start_point),
              view_angles=view_angles,
              speed=5,
              view_length=300,
              max_turn_angle=30,
              direction=start_direction)

    trained_agent = train_dqn(car)
