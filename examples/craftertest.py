import sys
sys.path.append("/home/mishaal/valuedream")
import gym
import crafter
import dreamerv2.api as dv2

config = dv2.defaults.update({
    'logdir': '~/logdir/crafter_multi_step',
    'log_every': 1e3,
    'train_every': 10,
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
    "steps": 1e7
}).parse_flags()

env = gym.make('CrafterReward-v1')  # Or CrafterNoReward-v1
env = crafter.Recorder(
  env, '~/logdir/crafter1',
  save_stats=True,
  save_video=False,
  save_episode=False,
)

dv2.train(env, config)

# obs = env.reset()
# done = False
# while not done:
#   action = env.action_space.sample()
#   obs, reward, done, info = env.step(action)