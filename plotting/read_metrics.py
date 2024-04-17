import collections
import json
import pathlib

import numpy as np

import common


def read_stats(indir, outdir, task, method, budget=int(1e6), verbose=False):
  indir = pathlib.Path(indir)
  outdir = pathlib.Path(outdir)
  runs = []
  print(f'Loading {method} {indir.name}...')
  filenames = sorted(list(indir.glob(f'**/{method}.jsonl')))
  for index, filename in enumerate(filenames):
    if not filename.is_file():
      continue
    rewards, lengths, achievements = load_stats(filename, budget)
    if sum(lengths) < budget - 1e4:
      message = f'Warning! Incomplete run ({sum(lengths)} < {budget} steps): '
      message += f'{filename.relative_to(indir.parent)}'
      print(f'==> {message}')
    runs.append(dict(
        task=task,
        method=method,
        seed=str(index),
        xs=np.cumsum(lengths).tolist(),
        reward=rewards,
        length=lengths,
        **achievements,
    ))
  
  # compute run scores
  from tqdm import tqdm
  for run in runs: 
    scores = []
    for i in tqdm(range(len(run["xs"]))):
      p = common.compute_success_rates([common.cut_run(run, i)], budget)[0]
      scores.append(common.compute_scores(p)[0][0])
    run["scores"] = scores
    
  if not runs:
    print('No completed runs.\n')
    return
  print_summary(runs, budget, verbose)
  outdir.mkdir(exist_ok=True, parents=True)
  filename = (outdir / f'{method}.json')
  filename.write_text(json.dumps(runs))
  print('Wrote', filename)
  print('')


def load_stats(filename, budget):
  steps = 0
  rewards = []
  lengths = []
  achievements = collections.defaultdict(list)
  for line in filename.read_text().split('\n'):
    if not line.strip():
      continue
    episode = json.loads(line)
    steps += episode['length']
    if steps > budget:
      break
    lengths.append(episode['length'])
    for key, value in episode.items():
      if key.startswith('achievement_'):
        achievements[key].append(value)
    unlocks = int(np.sum([(v[-1] >= 1) for v in achievements.values()]))
    health = -0.9
    rewards.append(unlocks + health)
  return rewards, lengths, achievements


def print_summary(runs, budget, verbose):
  episodes = np.array([len(x['length']) for x in runs])
  rewards = np.array([np.mean(x['reward']) for x in runs])
  lengths = np.array([np.mean(x['length']) for x in runs])
  percents, methods, seeds, tasks = common.compute_success_rates(
      runs, budget, sortby=0)
  scores = np.squeeze(common.compute_scores(percents))
  print(f'Score:        {np.mean(scores):10.2f} ± {np.std(scores):.2f}')
  print(f'Reward:       {np.mean(rewards):10.2f} ± {np.std(rewards):.2f}')
  print(f'Length:       {np.mean(lengths):10.2f} ± {np.std(lengths):.2f}')
  print(f'Episodes:     {np.mean(episodes):10.2f} ± {np.std(episodes):.2f}')
  if verbose:
    for task, percent in sorted(tasks, np.squeeze(percents).T):
      name = task[len('achievement_'):].replace('_', ' ').title()
      print(f'{name:<20}  {np.mean(percent):6.2f}%')


# read_stats(
#     indir='/Users/leeso/OneDrive - University of Toronto/Desktop/CSC/CSC4/project/stats',
#     outdir='score', task='', method='base16')
