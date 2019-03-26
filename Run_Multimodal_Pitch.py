from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import argparse
import random
import time
import sys
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import matplotlib.ticker as ticker

from matplotlib.ticker import MaxNLocator

import skimage.color, skimage.transform
import cv2
from vizdoom import *
np.set_printoptions(threshold=np.inf)


from pydub import AudioSegment
from playsound import playsound

from pydub.playback import play
import vizdoom as vzd


import skimage.color, skimage.transform
def MakeDir(path):
    try:
        os.makedirs(path)
    except:
        pass

lab = False
load_model = False
train = True
test_display = False
test_write_video = False
path_work_dir = "./"
vizdoom_path = "./"
vizdoom_scenario = vizdoom_path + "scenarios/Find_the_object_randomized_ver2.wad"

# Vizdoom parameters.

from Environment_Multimodal_Pitch import Environment_Multimodal_Pitch

learning_rate = 0.00025
discount_factor = 0.99
step_num = 600000
replay_memory_size = 10000
replay_memory_batch_size = 64

frame_repeat = 20
channels = 3
resolution = (40, 40) + (channels,) # Original: 480x640

channels_audio=1
resolution_samples=(1,114) + (channels_audio,)

start_eps = 1.0
end_eps = 0.1
eps_decay_iter = 0.33 * step_num

model_path = path_work_dir + 'Model/Model_Multimodal_20_Randomized_With_Walls_'+str(step_num)+'_Steps/'
save_each = 2000
step_load = 100

MakeDir(model_path)
model_name = model_path + "model"
# Global variables.
env = None

def Preprocess(img_pixel,img_audio):
    img_pixel = cv2.resize(img_pixel, (resolution[1], resolution[0]))
    img_pixel=np.reshape(img_pixel, resolution)
    
    img_audio=np.reshape(img_audio,list(resolution_samples))

    return img_pixel, img_audio

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

class ReplayMemory(object):
    def __init__(self, capacity):

        self.s1 = np.zeros((capacity,) + resolution, dtype=np.float32)#current state pixel
        self.s2 = np.zeros((capacity,) + resolution, dtype=np.float32)#next state pixel

        self.s3 = np.zeros((capacity,) + resolution_samples, dtype=np.float32) #current state audio
        self.s4 = np.zeros((capacity,) + resolution_samples, dtype=np.float32) #next state audio

        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def Add(self, s1, s3, action, s2, s4, isterminal, reward):

        self.s1[self.pos, ...] = s1
        self.s3[self.pos, ...] = s3
        self.a[self.pos] = action
        self.isterminal[self.pos] = isterminal
        if not isterminal:
                self.s2[self.pos, ...] = s2
                self.s4[self.pos, ...] = s4
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def Get(self, sample_size):

        i = random.sample(range(0, self.size-2), sample_size)
        return self.s1[i], self.s3[i], self.a[i], self.s2[i], self.s4[i], self.isterminal[i], self.r[i]

class Model(object):
    def __init__(self, session, actions_count):

        self.session = session
        # Create the input.
        self.s1_pixel = tf.placeholder(shape=[None] + list(resolution), dtype=tf.float32) #current state pixel
        self.s3_audio=tf.placeholder(shape=[None]+ list(resolution_samples),dtype=tf.float32) #current state audio
        self.q_ = tf.placeholder(shape=[None, actions_count], dtype=tf.float32)

        # Create the network for the pixels.
        conv1 = tf.contrib.layers.conv2d(self.s1_pixel, num_outputs=8, kernel_size=[3, 3], stride=[2, 2])
        conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=16, kernel_size=[3, 3], stride=[2, 2])
        conv2_flat = tf.contrib.layers.flatten(conv2)

        # Create the network for the audios.
        conv1_audio = tf.contrib.layers.conv2d(self.s3_audio, num_outputs=8, kernel_size=[3, 3], stride=[2, 2])
        conv2_audio = tf.contrib.layers.conv2d(conv1_audio, num_outputs=16, kernel_size=[3, 3], stride=[2, 2])
        conv2_flat_audio = tf.contrib.layers.flatten(conv2_audio)

        multimodal=tf.concat([conv2_flat,conv2_flat_audio],axis=1)

        fc1 = tf.contrib.layers.fully_connected(multimodal, num_outputs=128)
        self.q = tf.contrib.layers.fully_connected(fc1, num_outputs=actions_count, activation_fn=None)

        self.action = tf.argmax(self.q, 1)
        self.loss = tf.losses.mean_squared_error(self.q_, self.q)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)

    def Learn(self, state_pixel, state_audio, q):

        l, _ = self.session.run([self.loss, self.train_step], feed_dict={self.s1_pixel : state_pixel, self.s3_audio:state_audio, self.q_: q})
        return l

    def GetQ(self, state_pixel,state_audio):

        return self.session.run(self.q, feed_dict={self.s1_pixel : state_pixel,self.s3_audio:state_audio})

    def GetAction(self, state_pixel,state_audio):

        state_pixel = state_pixel.reshape([1] + list(resolution))#(1, 30, 45, 3)
        state_audio = state_audio.reshape([1] + list(resolution_samples))  # (1, 1, 100, 1)
        return self.session.run(self.action, feed_dict={self.s1_pixel: state_pixel,self.s3_audio:state_audio})[0]

class Agent(object):
    def __init__(self, num_actions):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

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
            s1, s3, a, s2, s4,isterminal, r = self.memory.Get(replay_memory_batch_size)
            # s1 is the current state using pixel information
            # s3 is the current state using audio information
            # s2 is the next state using pixel information
            # s4 is the next state using audio information
            q = self.model.GetQ(s1,s3)
            q2 = np.max(self.model.GetQ(s2,s4), axis=1)
            q[np.arange(q.shape[0]), a] = r + (1 - isterminal) * discount_factor * q2
            self.model.Learn(s1,s3,q)

    def GetAction(self, state,state_audio):
        if (random.random() <= 0.05):
            best_action = random.randint(0, self.num_actions-1)
        else:
            best_action = self.model.GetAction(state,state_audio)
        return best_action

    def Step(self, iteration):

        s1_pixel, s3_audio = env.Observation()
        s1, s3  = Preprocess(s1_pixel, s3_audio)

        # Epsilon-greedy.
        if (iteration < eps_decay_iter):
            eps = start_eps - iteration / eps_decay_iter * (start_eps - end_eps)
        else:
            eps = end_eps

        if (random.random() <= eps):
            best_action = random.randint(0, self.num_actions-1)
        else:
            best_action = self.model.GetAction(s1,s3)

        reward = env.Act(best_action, frame_repeat)
        self.rewards += reward

        isterminal = not env.IsRunning()

        if not isterminal:
            s2_pixel,s4_audio=env.Observation()
            s2, s4 = Preprocess(s2_pixel, s4_audio)
        else:
            s2=None
            s4=None
        self.memory.Add(s1, s3,best_action, s2, s4,isterminal, reward)
        self.LearnFromMemory()
    def Train(self):
        start_time = time.time()
        train_scores = []
        env.Reset()
        for step in range(1, step_num+1):
            self.Step(step)
            if(not env.IsRunning()):
                train_scores.append(self.rewards)
                self.rewards=0
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
        fps = 30.0  # / frame_repeat
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.cv.CV_FOURCC(*'XVID')
        out_video = cv2.VideoWriter(path_work_dir + "test.avi", fourcc, fps, size)

    reward_total = 0
    num_episodes = 10

    test=0
    while (num_episodes != 0):

        if (not env.IsRunning()):
            env.Reset()
            print("Total reward: {}".format(reward_total))
            reward_total = 0
            num_episodes -= 1

        state_raw_pixel,state_raw_audio = env.Observation()
        state_pixel,state_audio = Preprocess(state_raw_pixel,state_raw_audio)
        best_action=agent.GetAction(state_pixel,state_audio)

        for _ in range(frame_repeat):

            if (test_display):
                cv2.imshow("frame-test", state_raw_pixel)
                cv2.waitKey(20)

            if (test_write_video):
                out_video.write(state_raw_pixel)

            reward = env.Act(best_action, 1)
            reward_total += reward

            if (not env.IsRunning()):
                break

            state_raw_pixel,state_raw_audio = env.Observation()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="the GPU to use")
    args = parser.parse_args()

    if (args.gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    env = Environment_Multimodal_Pitch(vizdoom_scenario)
    agent = Agent(env.NumActions())
    if (train):
        agent.Train()

    Test(agent)
