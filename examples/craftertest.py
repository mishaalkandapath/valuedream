import sys
sys.path.append("/home/valuedream") #"/home/leesophie99/projects/valuedream") # "/home/mishaal/valuedream")
import gym
import crafter
import dreamerv2.api as dv2
import tensorflow as tf

config = dv2.defaults.update({
    'logdir': '~/logdir/crafter_testing',
    'log_every': 1e3,
    'train_every': 6,
    'prefill': 1e4,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
    "steps": 1e6
}).parse_flags()

env = gym.make('CrafterReward-v1')  # Or CrafterNoReward-v1
env = crafter.Recorder(
  env, '~/logdir/crafter_testings',
  save_stats=True,
  save_video=False,
  save_episode=False,
)

# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_visible_devices(gpus[1], "GPU")

dv2.train(env, config)

# obs = env.reset()
# done = False
# while not done:
#   action = env.action_space.sample()
#   obs, reward, done, info = env.step(action)