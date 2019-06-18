# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: Flood Sung
# Date: 2016.3.21
# -------------------------

import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from BrainDQN_Nature import BrainDQN
import numpy as np


def showThreshImg(observation):
	observation = np.rot90(observation) # 逆时针
	observation = observation[::-1] # x轴镜像，上下
	return observation

# preprocess raw image to 80*80 gray image
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	return np.reshape(observation,(80,80,1))

def playFlappyBird():
	# Step 1: init BrainDQN
	actions = 2
	brain = BrainDQN(actions)
	# Step 2: init Flappy Bird Game
	flappyBird = game.GameState()
	# Step 3: play game
	# Step 3.1: obtain init state
	action0 = np.array([1,0])  # do nothing
	observation0, reward0, terminal = flappyBird.frame_step(action0)

	observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
	
	brain.setInitState(observation0)

	# Step 3.2: run the game
	while 1!= 0:
		action = brain.getAction() # 对初始的状态有action反馈 at
		nextObservation,reward,terminal = flappyBird.frame_step(action) # 执行器获得指令，并输出该指令的奖励r(t),执行该指令导致的观测 o(t+1)，
		nextObservation = preprocess(nextObservation)
		
		tmp_img = showThreshImg(nextObservation) # 从flappyBird中得到的图像是一个旋转加镜像的
		cv2.imshow("process", tmp_img)
		cv2.waitKey(1)

		brain.setPerception(nextObservation,action,reward,terminal)

def main():
	playFlappyBird()

if __name__ == '__main__':
	main()