import numpy as np
import gym
import sys
import argparse

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge, LeakyReLU
from keras.optimizers import Adam

from rl.agents import ContinuousDQNAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor


class PendulumProcessor(Processor):
    def process_reward(self, reward):
        # The magnitude of the reward can be important. Since each step yields a relatively
        # high reward, we reduce the magnitude by two orders.
        return reward / 100.


def activation():
    return LeakyReLU(0.2)


ENV_NAME = 'Pendulum-v0'


def main(argv):
    default_nch = 64
    default_path = 'cdqn_{}_weights.h5f'.format(ENV_NAME)

    parser = argparse.ArgumentParser(description='Train CDQN on Pendulum-v0 environment.')
    parser.add_argument('--load', action="store_true", help='load pretrained model [default: train new model]')
    parser.add_argument('--visualize', action="store_true",
                        help='visualize during training [default: do not visualize]')
    parser.add_argument('--nch', action="store", default=default_nch,
                        help="hidden dimension [default: {}]".format(default_nch), type=int)
    parser.add_argument('--model', action="store", default=default_path,
                        help='path to save or load model [default: {}]'.format(default_path))
    args = parser.parse_args(argv)
    gym.undo_logger_setup()

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    nch = args.nch
    internal_model = Sequential(name="internal_model")
    internal_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    internal_model.add(Dense(nch))
    internal_model.add(activation())
    internal_model.add(Dense(nch))
    internal_model.add(activation())

    # Build all necessary models: V, mu, and L networks.
    V_model = Sequential(name="V_model")
    V_model.add(internal_model)
    V_model.add(Dense(nch))
    V_model.add(activation())
    V_model.add(Dense(1))
    print(V_model.summary())

    mu_model = Sequential(name="mu_model")
    mu_model.add(internal_model)
    mu_model.add(Dense(nch))
    mu_model.add(activation())
    mu_model.add(Dense(nb_actions))
    print(mu_model.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    observation_h = internal_model(observation_input)
    x = merge([action_input, observation_h], mode='concat')
    x = Dense(nch)(x)
    x = activation()(x)
    x = Dense(nch)(x)
    x = activation()(x)
    x = Dense(nch)(x)
    x = activation()(x)
    x = Dense(((nb_actions * nb_actions + nb_actions) / 2))(x)
    L_model = Model(input=[action_input, observation_input], output=x, name="L_model")
    print(L_model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    processor = PendulumProcessor()
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
    agent = ContinuousDQNAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                               memory=memory, nb_steps_warmup=100, random_process=random_process,
                               gamma=.99, target_model_update=1e-3, processor=processor)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    if args.load:
        # Load weights from a pretrained model
        agent.load_weights(args.model)
    else:
        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        agent.fit(env, nb_steps=50000, visualize=args.visualize, verbose=2, nb_max_episode_steps=200)

        # After training is done, we save the final weights.
        agent.save_weights(args.model, overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=200)


if __name__ == "__main__":
    main(sys.argv[1:])
