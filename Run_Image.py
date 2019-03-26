#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from skimage.transform import resize
import argparse
import random
import time
import sys
import os

import numpy as np
import cv2
import tensorflow as tf
import skimage.color, skimage.transform
def MakeDir(path):
    try:
        os.makedirs(path)
    except:
        pass

lab = False
load_model = False
train = True
test_display = True
test_write_video = False
path_work_dir = "./"
vizdoom_path = "./"
vizdoom_scenario = vizdoom_path + "scenarios/Find_the_object_randomized_ver2.wad"

# Vizdoom parameters.
from Environment_Image import EnvironmentImage

learning_rate = 0.00025
discount_factor = 0.99
step_num = 600000
replay_memory_size = 10000
replay_memory_batch_size = 64

frame_repeat = 20
channels = 3
resolution = (40, 40) + (channels,) # Original: 480x640

start_eps = 1.0
end_eps = 0.1
eps_decay_iter = 0.33 * step_num

model_path = path_work_dir + 'Model/Model_Image_Randomized_With_Walls_10_'+str(step_num)+'_Steps/'
save_each = 2000
step_load = 300

MakeDir(model_path)
model_name = model_path + "model"
env = None

def PrintStat(elapsed_time, step, step_num, train_scores):
    mean_train = 0
    std_train = 0
    min_train = 0
    max_train = 0
    if (len(train_scores) > 0):
        train_scores = np.array(train_scores)
        mean_train = train_scores.mean()
        std_train = train_scores.std()
        min_train = train_scores.min()
        max_train = train_scores.max()
    print("{}% | Steps: {}/{}, Episodes: {} Rewards: mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}".format(
        round((100.0 * step / step_num),2), step, step_num, len(train_scores), mean_train, std_train, min_train, max_train), file=sys.stderr)

def Preprocess(img):
    img = cv2.resize(img, (resolution[1], resolution[0]))
    return np.reshape(img, resolution)

class ReplayMemory(object):
    def __init__(self, capacity):

        self.s1 = np.zeros((capacity,) + resolution, dtype=np.float32)
        self.s2 = np.zeros((capacity,) + resolution, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def Add(self, s1, action, s2, isterminal, reward):

        self.s1[self.pos, ...] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, ...] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def Get(self, sample_size):

        i = random.sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

class Model(object):
    def __init__(self, session, actions_count):

        self.session = session

        # Create the input.
        self.s_ = tf.placeholder(shape=[None] + list(resolution), dtype=tf.float32)
        self.q_ = tf.placeholder(shape=[None, actions_count], dtype=tf.float32)

        # Create the network.
        conv1 = tf.contrib.layers.conv2d(self.s_, num_outputs=8, kernel_size=[3, 3], stride=[2, 2])
        conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=16, kernel_size=[3, 3], stride=[2, 2])
        conv2_flat = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128)

        self.q = tf.contrib.layers.fully_connected(fc1, num_outputs=actions_count, activation_fn=None)
        self.action = tf.argmax(self.q, 1)

        self.loss = tf.losses.mean_squared_error(self.q_, self.q)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)

    def Learn(self, state, q):

        state = state.astype(np.float32)
        l, _ = self.session.run([self.loss, self.train_step], feed_dict={self.s_: state, self.q_: q})
        return l

    def GetQ(self, state):

        state = state.astype(np.float32)
        return self.session.run(self.q, feed_dict={self.s_: state})

    def GetAction(self, state):

        state = state.astype(np.float32)
        state = state.reshape([1] + list(resolution))
        return self.session.run(self.action, feed_dict={self.s_: state})[0]

class Agent(object):

    def __init__(self, num_actions):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)

        self.model = Model(self.session, num_actions)
        self.memory = ReplayMemory(replay_memory_size)

        self.rewards = 0

        self.saver = tf.train.Saver(max_to_keep=1000)
        if (load_model):
            model_name_curr = model_name + "_{:04}".format(step_load)
            print("Loading model from: ", model_name_curr)
            self.saver.restore(self.session, model_name_curr)
        else:
            init = tf.global_variables_initializer()
            self.session.run(init)

        self.num_actions = num_actions

    def LearnFromMemory(self):

        if (self.memory.size > 2*replay_memory_batch_size):
            s1, a, s2, isterminal, r = self.memory.Get(replay_memory_batch_size)

            q = self.model.GetQ(s1)
            q2 = np.max(self.model.GetQ(s2), axis=1)
            q[np.arange(q.shape[0]), a] = r + (1 - isterminal) * discount_factor * q2
            self.model.Learn(s1, q)

    def GetAction(self, state):

        if (random.random() <= 0.05):
            a = random.randint(0, self.num_actions-1)
        else:
            a = self.model.GetAction(state)

        return a

    def Step(self, iteration):

        s1 = Preprocess(env.Observation())

        # Epsilon-greedy.
        if (iteration < eps_decay_iter):
            eps = start_eps - iteration / eps_decay_iter * (start_eps - end_eps)
        else:
            eps = end_eps

        if (random.random() <= eps):
            a = random.randint(0, self.num_actions-1)
        else:
            a = self.model.GetAction(s1)

        reward = env.Act(a, frame_repeat)
        self.rewards += reward

        isterminal = not env.IsRunning()
        s2 = Preprocess(env.Observation()) if not isterminal else None

        self.memory.Add(s1, a, s2, isterminal, reward)
        self.LearnFromMemory()

    def Train(self):

        print("Starting training.")
        start_time = time.time()
        train_scores = []
        env.Reset()
        for step in range(1, step_num+1):
            self.Step(step)
            if (not env.IsRunning()):
                train_scores.append(self.rewards)
                self.rewards = 0
                env.Reset()

            if (step % save_each == 0):
                model_name_curr = model_name + "_{:04}".format(int(step / save_each))
                self.saver.save(self.session, model_name_curr)

                PrintStat(time.time() - start_time, step, step_num, train_scores)

                train_scores = []

        env.Reset()

def Test(agent):
    if (test_write_video):
        size = (160, 120)
        fps = 30.0 
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.cv.CV_FOURCC(*'XVID')
        out_video = cv2.VideoWriter(path_work_dir + "test.avi", fourcc, fps, size)

    reward_total = 0
    num_episodes = 10
    while (num_episodes != 0):
        if (not env.IsRunning()):
            env.Reset()
            print("Total reward: {}".format(reward_total))
            reward_total = 0
            num_episodes -= 1

        state_raw = env.Observation()

        state = Preprocess(state_raw)
        action = agent.GetAction(state)

        for _ in range(frame_repeat):
            # Display.
            if (test_display):
                cv2.imshow("frame-test", state_raw)
                cv2.waitKey(20)

            if (test_write_video):
                out_video.write(state_raw)

            reward = env.Act(action, 1)
            reward_total += reward

            if (not env.IsRunning()):
                break

            state_raw = env.Observation()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="the GPU to use")
    args = parser.parse_args()

    if (args.gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    env = EnvironmentImage(vizdoom_scenario)

    agent = Agent(env.NumActions())
    if(train):
        agent.Train()
    Test(agent)
