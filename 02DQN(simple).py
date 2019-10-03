from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import random
from datetime import datetime


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
        self.batch_size = 32
        self.train_start = 50
        self.update_target_rate = 100
        self.discount_factor = 0.95
        self.no_op_steps = 30
        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.avg_q_max, self.avg_loss = 0, 0

        if self.load_model:
            self.model.load_weights("./dqn3.h5")
            print("weight load!")

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

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), True
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0]), False

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

    for e in range(4):
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

        while not done:
            # if agent.render:
            env.render()
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
            # print(info)
            # 각 타임스텝마다 상태 전처리
            next_state = agent.pre_processing(observe)
            next_state = np.reshape([next_state], (1, 240, 256, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)
            agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0])

            real_reward = reward

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
                print(
                    "episode : {}, score : {}, step : {}, avg q : {}, avg loss : {}".format(
                        e, score, agent.epsilon, global_step, agent.avg_q_max / float(step), agent.avg_loss / float(step)
                    )
                )
                print("epsilon : {}, greedy : {}".format(count_epsilon, count_greedy))
                print()

                # if e < 2:
                #     pass
                # else:
                print("time elapsed : {} sec".format((datetime.now() - global_start).seconds))
                global_start = datetime.now()
                print()
                print()

                agent.avg_q_max, agent.avg_loss, global_step = 0, 0, 0


if __name__ == "__main__":
    main()
