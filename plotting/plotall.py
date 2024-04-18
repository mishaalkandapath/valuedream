from read_metrics import read_stats
from plot_counts import plot_counts
from plot_reward import plot_reward
from plot_scores import plot_scores_hist, plot_scores_time
from plot_spectrum import plot_spectrum

# modify
STATS_DIR = "experiment_results"
METHODS = ["1", "base8", "base16", "itervaml16", "mstep_itervaml8", "mstep8", 
           "mstep_backpropwm8", "mstep_itervaml16", "mstep16"]


## keep
MAPPINGS = {
    '1':["1", '#00323a'], 
    'base8': ['8 DreamerV2', '#0d9937'], 
    'backpropwm8':["8 Backprop World Model", '#ba9872'], 
    'itervaml8':['8 iterVAML', '#f6bf1d'], 
    'mstep8': ['8 MultiStep', '#26cdd6'], 
    'mstep_backpropwm8':["8 MultiStep Backprop to WM", '#1958ea'], 
    'mstep_itervaml8': ["8 MultiStep iterVAML", '#cc1d0a'],
    
    'base16': ['16 DreamerV2', '#63f78f'], 
    'backpropwm16':["16 Backprop to WM", '#eddfc9'], 
    'itervaml16':['16 iterVAML', '#f2d974'], 
    'mstep16': ['16 MultiStep', '#82f4fa'], 
    'mstep_backpropwm16':["16 MultiStep Backprop to WM", '#8fa5f0'], 
    'mstep_itervaml16': ["16 MultiStep iterVAML", '#ff7575']
}

PLOT_DIR = 'plots'
OUTDIR = 'score'
inpaths = [f'{OUTDIR}/{x}.json' for x in METHODS]
legend = {x:MAPPINGS[x][0] for x in METHODS}
colors = [MAPPINGS[x][1] for x in METHODS]


if __name__ == "__main__":
    # generate stats
    for x in ["mstep_backpropwm8", "mstep_itervaml16", "mstep16"]: 
        read_stats(indir=STATS_DIR, outdir=OUTDIR, task='', method=x)
    
    # plot
    plot_counts(inpaths[0], f'{PLOT_DIR}/{METHODS[0]}_counts.pdf', colors[0], budget=670000) #654604
    for i in range(1, len(METHODS)): plot_counts(inpaths[i], f'{PLOT_DIR}/{METHODS[i]}_counts.pdf', colors[i])
    plot_reward(inpaths, f'{PLOT_DIR}/reward.pdf', legend, colors)
    plot_scores_hist(inpaths, f'{PLOT_DIR}/scores_hist.pdf', legend, colors, ylim=12)
    plot_scores_time(inpaths, f'{PLOT_DIR}/scores_time.pdf', legend, colors)
    plot_spectrum(inpaths, f'{PLOT_DIR}/spectrum-reward.pdf', legend, colors)


