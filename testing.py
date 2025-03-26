import gymnasium as gym
import numpy as np
from gymnasium.wrappers.record_video import RecordVideo

q_values = np.load("q_values.npy")
episodes = 5
env = gym.make("Taxi-v3", render_mode = "rgb_array")
env = RecordVideo(env, video_folder = "taxi-agent", name_prefix = "eval", episode_trigger=lambda x:True)
epsisode_rewards = []
trigger = lambda t: t % 10 == 0

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(q_values[state, :])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = next_state
        total_reward += reward
    epsisode_rewards.append(total_reward)

env.close()

print("testing completed.")
print(epsisode_rewards)