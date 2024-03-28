import crafter, gym, numpy
env = gym.make('CrafterReward-v1')  # Or CrafterNoReward-v1
env = crafter.Recorder(
  env, './path/to/logdir',
  save_stats=True,
  save_video=False,
  save_episode=False,
)

obs = env.reset()
num_steps = 3
for step in range(num_steps):
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)

def get_intersection_from_sequence(seq: list) -> numpy.ndarray:
  pass # lol