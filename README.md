# 🚕 Taxi-v3 – Q-Learning Agent

This project implements a **Q-learning agent** for the classic `Taxi-v3` environment from [Gymnasium](https://gymnasium.farama.org/).  
The goal is to teach an agent how to efficiently pick up and drop off passengers at the correct locations in a gridworld environment.

---

## 🌍 Environment Overview

- **Environment:** `Taxi-v3`
- **States:** 500 discrete states (5×5 grid × 5 locations × 4 destinations)
- **Actions:**  
  `0 = south`, `1 = north`, `2 = east`, `3 = west`,  
  `4 = pickup`, `5 = dropoff`
- **Rewards:**
  - `-1` per timestep
  - `+20` for successful drop-off
  - `-10` for illegal pickup/dropoff

---

## 🧠 Algorithm

- **Learning type:** Tabular Q-learning
- **Exploration:** Epsilon-greedy with exponential decay
- **Update rule:**  
  Q(s, a) -> Q(s, a) + alpha [r + gamma * max Q(s', a) - Q(s, a)]
- **Policy:** Greedy for evaluation

---

## ⚙️ Hyperparameters


- Learning rate   | `alpha = 0.5` 
- Discount factor | `gamma = 0.99` 
- Initial epsilon | `1.0` 
- Minimum epsilon | `0.1` 
- Decay rate      | Adaptively computed 
- Episodes        | `50,000` 

---

## 📁 Project Structure

- `training.py`  | Trains the Q-learning agent and saves the `q_values.npy` if average reward improves 
- `testing.py`   | Loads the saved Q-table, evaluates the agent for 5 episodes, and records videos 
- `q_values.npy` | (Generated) Learned Q-values after training 
- `taxi-agent/`  | (Generated) Folder containing evaluation videos 

---

##Note

I used a Gymnasium wrapper `RecordVideo` to record the episodes of my agent. Since this is optional, you can remove it if you want to. 
