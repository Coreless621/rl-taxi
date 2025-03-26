import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make("Taxi-v3")
num_states = env.observation_space.n
num_actions = env.action_space.n

q_values = np.zeros((num_states, num_actions))
best_average_reward = 2.17

# hyperparameters
episodes = 50_000
alpha = 0.5
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.1
decay = (min_epsilon/epsilon) ** (1/episodes) # calculating decay rate 

episode_rewards = []
for episode in tqdm(range(episodes)):
    done = False
    state, info = env.reset()
    total_reward = 0
    
    while not done:
        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_values[state, :])

        next_state, reward, terminated, truncated, info = env.step(action)

        #q_values update
        q_values[state, action] = q_values[state, action] + alpha * (reward + gamma * np.max(q_values[next_state, :]) - q_values[state, action])
        
        done = terminated or truncated
        total_reward += reward
        state = next_state

        #epsilon decay
        epsilon = max(min_epsilon, epsilon * decay)
    episode_rewards.append(total_reward)

print("training completed.")

if np.mean(episode_rewards[-100:]) > best_average_reward:
    np.save("q_values.npy", q_values)
    print(f"New record average reward over last 100 episodes!: {np.mean(episode_rewards[-100:]):.2f}")
    print("New optimal Q values saved.")


plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Average Performance over time")
plt.show()

