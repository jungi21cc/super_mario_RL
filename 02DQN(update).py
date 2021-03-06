from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
from datetime import datetime
from joblib import dump, load
import os.path
import os
import math
import gc

### slcak
from slacker import Slacker
from config import token


def slack_msg(msg):
    slack = Slacker(token)
    attachments_dict = dict()
    attachments_dict["pretext"] = "{}".format(datetime.now())
    attachments_dict["title"] = "=====DQN result====="
    attachments_dict["text"] = "```{}```".format(msg)
    attachments_dict["mrkdwn_in"] = ["text", "pretext"]  # 마크다운을 적용시킬 인자들을 선택합니다.
    attachments = [attachments_dict]
    slack.chat.post_message(channel="#jarvis", text=None, attachments=attachments)


class DQNAgent:
    def __init__(self, action_size=7):
        # self.render = False
        self.load_model = True
        self.load_memory = False
        # 상태와 행동의 크기 정의
        self.state_size = (240, 256, 4)
        self.action_size = action_size
        # DQN 하이퍼파라미터
        self.epsilon = 0.115
        # self.epsilon_min = 0.1
        # # self.exploration_steps = 1000.
        self.epsilon_decay_step = 0.0005
        self.batch_size = 32
        self.train_start = 50
        self.update_target_rate = 100
        self.discount_factor = 0.925
        # 리플레이 메모리, 최대 크기 400000
        self.memory = deque(maxlen=100000)
        self.no_op_steps = 30
        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()
        self.avg_q_max, self.avg_loss = 0, 0

        if self.load_model:
            self.model.load_weights("./dqn.h5")
            print("weight load!")

        # if os.path.exists("memory.joblib"):
        #     if self.load_memory:
        #         self.memory = load("./memory.joblib")
        #         print("memory load!")
            # else:
            #     pass

    # Huber Loss를 이용하기 위해 최적화 함수를 직접 정의
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = Adam()
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(64, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(16, (2, 2), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    # 타겟 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), True
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0]), False

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        target_value = self.target_model.predict(next_history)

        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * np.amax(target_value[i])

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    # 학습속도를 높이기 위해 흑백화면으로 전처리
    def pre_processing(self, observe):
        processed_observe = np.uint8(resize(rgb2gray(observe), (240, 256), mode='constant') * 255)
        return processed_observe


def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    agent = DQNAgent(action_size=7)

    scores, episodes, global_step = [], [], 0

    global_start = datetime.now()
    local_start = datetime.now()

    print()
    print("=" * 100)
    print("RL environment initialized")
    print("=" * 100)
    print()
    gc.collect()

    for e in range(1000):
        e = e + 1
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        observe = env.reset()

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        state = agent.pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 240, 256, 4))

        count_epsilon = 0
        count_greedy = 0

        coinStatus = 0
        marioStatus = "small"
        flagStatus = False
        softReward = 0
        lifeStatus = 2

        while not done:
            # if agent.render:
            #     env.render()
            global_step += 1
            step += 1
            # 바로 전 4개의 상태로 행동을 선택
            action, res = agent.get_action(history)
            if res:
                count_epsilon += 1
            else:
                count_greedy += 1

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observe, reward, done, info = env.step(action)
            # 각 타임스텝마다 상태 전처리
            next_state = agent.pre_processing(observe)
            next_state = np.reshape([next_state], (1, 240, 256, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)
            agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0])
            if start_life > info['life']:
                dead = True
                start_life = info['life']
            # reward = np.clip(reward, -1., 1.)
            real_reward = reward

            ###
            ###
            ###
            # reward = reward
            # if coinStatus != info["coins"]:
            #     coinStatus = info["coins"]
            #     reward = reward + 10
            # if marioStatus != info["status"]:
            #     marioStatus = info["status"]
            #     reward = reward + 200
            # if flagStatus != info["flag_get"]:
            #     flagStatus = info["flag_get"]
            #     reward = reward + 200
            # if lifeStatus != info["life"]:
            #     lifeStatus = info["life"]
            #     reward = reward - 20
            #
            # if info["x_pos"] < 10:
            #     info["x_pos"] = 10
            # if info["time"] < 10:
            #     info["time"] = 10
            #
            # reward = reward + math.log((info["x_pos"] / info["time"]) + info["x_pos"])

            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, dead)
            if len(agent.memory) >= agent.train_start:
                agent.train_model()
            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            # score += reward
            score += real_reward

            if dead:
                dead = False
            else:
                history = next_history

            if global_step == 0:
                pass
            elif global_step % 1000 == 0:
                print("local step : {}, time : {} sec, epsilon : {}".format(global_step, (datetime.now() - local_start).seconds, agent.epsilon))
                local_start = datetime.now()

            if done:
                ep_result = "episode : {}, score : {}, memory : {}, step : {}".format(e, score, len(agent.memory), global_step)
                print(ep_result)
                print("epsilon : {}, greedy : {}".format(count_epsilon, count_greedy))
                print()
                print("time elapsed : {} sec".format((datetime.now() - global_start).seconds))
                global_start = datetime.now()
                agent.epsilon = agent.epsilon - agent.epsilon_decay_step
                print("epsilon decay to {}!".format(agent.epsilon))
                print()

                slack_msg(ep_result)

                # if score > 2000 and score <= 3000:
                #     agent.epsilon = 0.075
                # elif score > 3000 and score <= 5000:
                #     agent.epsilon = 0.05
                # elif score > 5000 and score <= 10000:
                #     agent.epsilon = 0.005

                agent.avg_q_max, agent.avg_loss, global_step = 0, 0, 0

        # 1000 에피소드마다 모델 저장
        if e == 0:
            pass
        elif e % 2 == 0:
            agent.model.save_weights("./dqn.h5")
            # dump(agent.memory, "memory.joblib")
            print("model saved!")
            print()

        gc.collect()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        slack_msg(e)

