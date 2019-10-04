from keras.optimizers import RMSprop
from keras import backend as K
import random
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from tqdm import tqdm
from datetime import datetime

### slcak
from slacker import Slacker
from config import token

import gc

global episode
episode = 0
EPISODES = 8000000


def slack_msg(msg):
    slack = Slacker(token)
    attachments_dict = dict()
    attachments_dict["pretext"] = "{}".format(datetime.now())
    attachments_dict["title"] = "=====A2C result====="
    attachments_dict["text"] = "```{}```".format(msg)
    attachments_dict["mrkdwn_in"] = ["text", "pretext"]  # 마크다운을 적용시킬 인자들을 선택합니다.
    attachments = [attachments_dict]
    slack.chat.post_message(channel="#jarvis", text=None, attachments=attachments)


class A2C:
    def __init__(self, action_size):
        self.state_size = (240, 256, 4)
        self.action_size = action_size
        self.states, self.actions, self.rewards = [], [], []

        self.discount_factor = 0.95
        self.no_op_steps = 30

        self.actor, self.critic = self.build_model()
        # self.actor_optimize = self.actor_optimizer()
        # self.critic_optimize = self.critic_optimizer()

        self.load_model = True
        self.pre_fix = "a2c"

        if self.load_model:
            self.actor.load_weights("./a2c_actor.h5")
            self.critic.load_weights("./a2c_critic.h5")
            print("weight load!")

    def build_model(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(64, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Conv2D(16, (2, 2), strides=(1, 1), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(128, activation='relu')(conv)
        fc = Dense(64, activation='relu')(fc)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor.summary()
        critic.summary()

        return actor, critic

    # 정책신경망의 출력을 받아서 확률적으로 행동을 선택
    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.actor.predict(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    # 샘플을 저장
    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    # 로컬신경망을 글로벌신경망으로 업데이트
    def update_model(self):
        self.actor.set_weights(self.actor.get_weights())
        self.critic.set_weights(self.critic.get_weights())

    # k-스텝 prediction 계산
    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            running_add = self.critic.predict(np.float32(self.states[-1] / 255.))[0]

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

   # 정책신경망과 가치신경망을 업데이트
    def train_model(self):
        # print("discount prediction!")
        # discounted_prediction = self.discounted_prediction(self.rewards, done)

        # states = np.zeros((len(self.states), 240, 256, 4))
        # for i in tqdm(range(len(self.states))):
        #     states[i] = self.states[i]
        #
        # states = np.float32(states / 255.)
        #
        # values = self.critic.predict(states)
        # values = np.reshape(values, len(values))

        # advantages = discounted_prediction - values

        gc.collect()

        # print("actor optimizer!")
        self.actor_optimizer()
        # print("critic optimizer!")
        self.critic_optimizer()
        # print("optimize done!")
        self.states, self.actions, self.rewards = [], [], []

    def save_model(self):
        self.actor.save_weights(self.pre_fix + "_actor.h5")
        self.critic.save_weights(self.pre_fix + "_critic.h5")
        print("model saved!")

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        # 정책 크로스 엔트로피 오류함수
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        # 탐색을 지속적으로 하기 위한 엔트로피 오류
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        # 두 오류함수를 더해 최종 오류함수를 만듬
        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop()
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages], [loss], updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # [반환값 - 가치]의 제곱을 오류함수로 함
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = RMSprop()
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_prediction], [loss], updates=updates)
        return train


def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(
        resize(rgb2gray(processed_observe), (240, 256), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    # env = gym.make(env_name)
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    agent = A2C(action_size=7)
    # agent.load_model("a3c_actor.h5")

    step = 0

    global_start = datetime.now()
    local_start = datetime.now()

    gc.collect()

    for e in range(10):
        e = e + 1
        print("EPISODE START {}".format(e))
        done = False
        dead = False

        score, start_life = 0, 5
        observe = env.reset()
        next_observe = observe

        for _ in range(random.randint(1, 20)):
            observe = next_observe
            next_observe, _, _, _ = env.step(1)

        state = pre_processing(next_observe, observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 240, 256, 4))

        while not done:
            # env.render()
            step += 1
            observe = next_observe

            action, policy = agent.get_action(history)

            # if action == 1:
            #     fake_action = 2
            # elif action == 2:
            #     fake_action = 3
            # else:
            #     fake_action = 1
            #
            # if dead:
            #     fake_action = 1
            #     dead = False

            next_observe, reward, done, info = env.step(action)

            next_state = pre_processing(next_observe, observe)
            next_state = np.reshape([next_state], (1, 240, 256, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_life > info['life']:
                dead = True
                reward = -1
                start_life = info['life']

            score += reward

            agent.append_sample(history, action, reward)

            if dead:
                history = np.stack((next_state, next_state, next_state, next_state), axis=2)
                history = np.reshape([history], (1, 240, 256, 4))
            else:
                history = next_history

            if step % 100:
                agent.update_model()
                # print("soft update!")
            if step == 0:
                pass
            elif step % 1000 == 0:
                print("local step : {}, time : {} sec".format(step, (datetime.now() - local_start).seconds))
                local_start = datetime.now()

            gc.collect()

        # if done, plot the score over episodes
        if done:
            # episode += 1
            ep_res = "episode: {},  score: {}, step: {}".format(e, score, step)
            # print("episode:", e, "  score:", score, "  step:", step)
            step = 0
            agent.train_model()
            agent.save_model()
            print("time elapsed : {} sec".format((datetime.now() - global_start).seconds))
            global_start = datetime.now()

            slack_msg(ep_res)

