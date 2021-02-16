import numpy as np
import gym
from DQNAgent import DQNAgent

env = gym.make("CartPole-v1")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

agent = DQNAgent(observation_space, action_space)
runs = 0

while True:
  state = np.reshape(env.reset(), [1, observation_space])
  runs += 1
  step = 0

  while True:
    step += 1
    #env.render()

    action = agent.action(state) #agent decides action to take
    next_state, reward, done, info = env.step(action)

    if done:
      reward *= -1
    
    next_state = np.reshape(next_state, [1, observation_space])
    agent.save_memory(state, action, reward, next_state, done)

    state = next_state

    if done:
      print(f"Run: {runs}, explore_rate: {agent.explore_rate}, score: {step}")
      break

    agent.replay()

# for i_episode in range(20):
#   observation = env.reset() #reset env when "game is over", observation is state
  
#   for t in range(100):
#     env.render()
#     print(observation)
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)

#     if done:
#       print(f"Episode finished after {t + 1} timesteps")
#       break

# env.close()