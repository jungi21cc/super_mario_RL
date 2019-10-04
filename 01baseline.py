from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def main():

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    done = False

    for e in range(100):
        state = env.reset()

        while not done:
            env.render()
            state, reward, done, info = env.step(env.action_space.sample())

    env.close()


if __name__ == "__main__":
    main()