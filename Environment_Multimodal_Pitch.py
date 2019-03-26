from __future__ import division
from __future__ import print_function

import itertools as it
import numpy as np
np.set_printoptions(threshold=np.inf)
from vizdoom import *
from pydub import AudioSegment
from pydub.playback import play
import cv2
import scipy
import random

def Extract_Pitch(self,player_pos_x,player_pos_y, object_pox_x,object_pos_y):
    import librosa
    from scipy.io.wavfile import read as read_wav
    import os
    import numpy as np
    sampling_rate, data = read_wav("Test.wav")
    y, sr = librosa.load('Test.wav', sr=sampling_rate)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=1600)

    player = [player_pos_x, player_pos_y]
    target = [object_pox_x, object_pos_y]

    distance = scipy.spatial.distance.euclidean(player, target)

    if (distance == 0):
        factor = 1000
    else:
        factor = 1000/ distance

    pitch_values = []

    for i in range(len(pitches[0])):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]+ factor
        pitch_values.append(pitch)
    return np.array(pitch_values)

class Environment_Multimodal_Pitch(object):
    def __init__(self, scenario_path):
        print("Initializing doom.")
        self.game = DoomGame()
        self.game.set_doom_scenario_path(scenario_path)
        self.game.set_doom_map("map01")
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_render_hud(False) # False
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        self.game.add_available_button(Button.TURN_LEFT)
        self.game.add_available_button(Button.TURN_RIGHT)
        self.game.add_available_button(Button.MOVE_FORWARD)
        self.game.add_available_button(Button.MOVE_BACKWARD)
        self.game.set_episode_timeout(1000)
        self.game.set_episode_start_time(14)
        self.game.set_window_visible(False)
        self.game.set_sound_enabled(False)
        self.game.set_living_reward(-1)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_labels_buffer_enabled(True)
        self.game.clear_available_game_variables()
        self.game.add_available_game_variable(GameVariable.POSITION_X)
        self.game.add_available_game_variable(GameVariable.POSITION_Y)
        self.game.add_available_game_variable(GameVariable.POSITION_Z)
        self.game.add_available_game_variable(GameVariable.AMMO2)
        self.game.init()
        print("Doom initialized.")

        n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=n)]
        self.num_actions = len(self.actions)
        print(self.num_actions)

    def NumActions(self):
        return self.num_actions

    def Reset(self):
        self.game.new_episode()

    def Act(self, action, frame_repeat):
        action = self.MapActions(action)
        return self.game.make_action(self.actions[action], frame_repeat)

    def IsRunning(self):
        return (not self.game.is_episode_finished())

    def Observation(self):

        goal_position_x = self.game.get_game_variable(GameVariable.USER1)
        goal_position_y = self.game.get_game_variable(GameVariable.USER2)

        goal_position_x = doom_fixed_to_double(goal_position_x)
        goal_position_y = doom_fixed_to_double(goal_position_y)

        target_position_x = goal_position_x
        target_position_y = goal_position_y

        player_position_x= self.game.get_game_variable(GameVariable.POSITION_X)
        player_position_y= self.game.get_game_variable(GameVariable.POSITION_Y)

        pixel_data = self.game.get_state().screen_buffer
        return pixel_data, Extract_Pitch(self, player_position_x, player_position_y,target_position_x,target_position_y)

    def MapActions(self, action_raw):
        return action_raw
