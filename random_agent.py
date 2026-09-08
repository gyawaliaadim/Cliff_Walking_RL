import gymnasium as gym
import numpy as np

cliffEnv = gym.make("CliffWalking-v1",render_mode="rgb_array")

done=False
obs, info = cliffEnv.reset(seed=123,options={})


step=0
while not done:
    action = cliffEnv.action_space.sample()
    print(obs,"-->",action)
    # if step%250==0:
    frame = cliffEnv.render()
    print(frame.shape)
    obs, reward, terminated, truncated, info = cliffEnv.step(action)
    done = terminated or truncated
    step+=1
cliffEnv.close()