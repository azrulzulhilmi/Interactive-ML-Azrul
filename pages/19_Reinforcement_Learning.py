import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Reinforcement Learning", page_icon="🤖", layout="wide")
from utils import load_css
load_css()
st.title("🤖 Reinforcement Learning (Q-Learning)")
st.markdown("""
**Reinforcement Learning** involves an agent learning to make decisions by taking actions in an environment to maximize a cumulative reward.
**Q-Learning** is a model-free algorithm where the agent builds a "Q-Table", learning the expected future reward for taking a specific action in a specific state.
""")

st.sidebar.header("Agent Parameters")
learning_rate = st.sidebar.slider("Learning Rate (Alpha)", 0.01, 1.0, 0.1, 0.05)
discount_factor = st.sidebar.slider("Discount Factor (Gamma)", 0.1, 0.99, 0.9, 0.05)
exploration_rate = st.sidebar.slider("Exploration Rate (Epsilon)", 0.01, 1.0, 0.2, 0.05)
episodes = st.sidebar.slider("Training Episodes", 10, 500, 100, 10)

# A simple 1D Grid World [S, 0, 0, 0, G]
# S=Start, G=Goal (Reward=1), others=0. State is index 0 to 4. Actions: 0=Left, 1=Right
n_states = 5
n_actions = 2
q_table = np.zeros((n_states, n_actions))

rewards_history = []

for ep in range(episodes):
    state = 0 # Start at leftmost
    total_reward = 0
    done = False
    
    while not done:
        # Epsilon-greedy action
        if np.random.uniform(0, 1) < exploration_rate:
            action = np.random.choice([0, 1])
        else:
            action = np.argmax(q_table[state, :])
            
        # Step in environment
        if action == 0: # Left
            next_state = max(0, state - 1)
        else: # Right
            next_state = min(n_states - 1, state + 1)
            
        # Get Reward
        reward = 1 if next_state == 4 else 0
        done = (next_state == 4)
        
        # Update Q-Table
        best_next_action = np.argmax(q_table[next_state, :])
        td_target = reward + discount_factor * q_table[next_state, best_next_action]
        td_error = td_target - q_table[state, action]
        q_table[state, action] += learning_rate * td_error
        
        state = next_state
        total_reward += reward
        
    rewards_history.append(total_reward)

col1, col2 = st.columns([3, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(q_table.T, cmap='YlGn', aspect='auto')
    ax.set_xticks(range(n_states))
    ax.set_xticklabels(['Start (0)', '1', '2', '3', 'Goal (4)'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Left', 'Right'])
    ax.set_xlabel('State')
    ax.set_ylabel('Action')
    ax.set_title('Learned Q-Table Values')
    
    # Add text annotations
    for i in range(n_states):
        for j in range(n_actions):
            text = ax.text(i, j, f"{q_table[i, j]:.2f}", ha="center", va="center", color="black")
            
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Training Info")
    st.write(f"Trained for **{episodes}** episodes.")
    st.write("Higher values in the table indicate the agent expects more reward by taking that action in that state.")
    
    # Extract optimal policy
    policy = ["Left" if np.argmax(q_table[s]) == 0 else "Right" for s in range(n_states-1)]
    st.write("**Optimal Policy Found:**")
    for s, act in enumerate(policy):
        st.write(f"State {s} -> Go {act}")

st.markdown("---")
with st.expander("View Python Code"):
    st.code("""
import numpy as np

# Initialize Q-table
q_table = np.zeros((5, 2)) # 5 states, 2 actions (Left/Right)

# Q-Learning Update Rule
for episode in range(100):
    state = 0
    done = False
    
    while not done:
        # Choose action (simplified, normally use epsilon-greedy)
        action = np.argmax(q_table[state, :]) 
        
        # Take action, get next state and reward
        next_state = state + 1 if action == 1 else state - 1
        reward = 1 if next_state == 4 else 0
        done = (next_state == 4)
        
        # Bellman Equation Update
        best_future_q = np.max(q_table[next_state, :])
        q_table[state, action] = q_table[state, action] + 0.1 * (reward + 0.9 * best_future_q - q_table[state, action])
        
        state = next_state
    """, language="python")