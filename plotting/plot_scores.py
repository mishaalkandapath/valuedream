import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import common

def plot_scores_hist(inpaths, outpath, legend, colors, budget=1e6, ylim=None):
  runs = common.load_runs(inpaths, budget)
  percents, methods, seeds, tasks = common.compute_success_rates(runs, budget)
  scores = common.compute_scores(percents)
  if not legend:
    methods = sorted(set(run['method'] for run in runs))
    legend = {x: x.replace('_', ' ').title() for x in methods}
  legend = dict(reversed(legend.items()))

  scores = scores[np.array([methods.index(m) for m in legend.keys()])]
  mean = np.nanmean(scores, -1)
  std = np.nanstd(scores, -1)

  fig, ax = plt.subplots(figsize=(4, 3))
  centers = np.arange(len(legend))
  width = 0.7
  colors = list(reversed(colors[:len(legend)]))
  error_kw = dict(capsize=5, c='#000')
  ax.bar(centers, mean, yerr=std, color=colors, error_kw=error_kw)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.tick_params(
      axis='x', which='both', width=50, length=0.8, direction='inout')
  ax.set_xlim(centers[0] - 2 * (1 - width), centers[-1] + 2 * (1 - width))
  ax.set_xticks(centers + 0.0)
  ax.set_xticklabels(
      list(legend.values()), rotation=45, ha='right', rotation_mode='anchor')

  ax.set_ylabel('Crafter Score (%)')
  if ylim:
    ax.set_ylim(0, ylim)

  fig.tight_layout()
  pathlib.Path(outpath).parent.mkdir(exist_ok=True, parents=True)
  fig.savefig(outpath)
  print(f'Saved {outpath}')


def plot_scores_time(inpaths, outpath, legend, colors, cols=4, budget=1e6):
  runs = common.load_runs(inpaths, budget)
  percents, methods, seeds, tasks = common.compute_success_rates(runs, budget)
  
  if not legend:
    methods = sorted(set(run['method'] for run in runs))
    legend = {x: x.replace('_', ' ').title() for x in methods}
  borders = np.arange(0, budget, 1e4)
  
  basedreamermax = -1
  fig, ax = plt.subplots(figsize=(4.5, 2.3))
  for j, (method, label) in enumerate(legend.items()):
    relevant = [run for run in runs if run['method'] == method]
    if not relevant:
      print(f'No runs found for method {method}.')
    # Average within each time bucket.
    binned_xs, binned_ys = [], []
    for run in relevant:
      xs, ys = common.binning(run['xs'], run['scores'], borders, np.nanmean)
      binned_xs.append(xs)
      binned_ys.append(ys)
    xs = np.concatenate(binned_xs)
    ys = np.concatenate(binned_ys)
    maxs = max(common.binning(xs, ys, borders, np.nanmax)[1])
    if "base" in method and maxs > basedreamermax: basedreamermax = maxs
    # Compute mean and stddev over seeds.
    means = common.binning(xs, ys, borders, np.nanmean)[1]
    stds = common.binning(xs, ys, borders, np.nanstd)[1]
    # Plot line and shaded area.
    kwargs = dict(alpha=0.2, linewidths=0, color=colors[j], zorder=10 - j)
    ax.fill_between(borders[1:], means - stds, means + stds, **kwargs)
    ax.plot(borders[1:], means, label=label, color=colors[j], zorder=100 - j)
  
  # add line for base dreamer
  ax.axhline(y=basedreamermax, c='#888888', ls='--', lw=1)
  ax.text(0.03e6, basedreamermax+.4, 'DreamverV2 best', c='#888888',fontsize="x-small")
  # ax.set_ylim(0, 12)

  ax.set_title('Crafter Scores')
  ax.set_xlim(0, budget)
  ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
  ax.grid(alpha=0.3)
  ax.xaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
  ax.yaxis.set_major_locator(ticker.MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))

  fig.tight_layout(rect=(0, 0, 0.55, 1))
  fig.legend(bbox_to_anchor=(0.52, 0.54), loc='center left', frameon=False)

  pathlib.Path(outpath).parent.mkdir(exist_ok=True, parents=True)
  fig.savefig(outpath)
  print(f'Saved {outpath}')


