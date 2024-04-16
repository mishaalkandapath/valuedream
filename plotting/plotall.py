from read_metrics import read_stats
from plot_counts import plot_counts
from plot_reward import plot_reward
from plot_scores import plot_scores
from plot_spectrum import plot_spectrum

## keep
MAPPINGS = {
    'base': ['DreamerV2', '#595457'], 
    'itervaml':['iterVAML', '#9e1946'], 
    '1':["1", '#710627'], # 'value_latent':["Value-based Latents", '#710627'], 
    'mstep': ['MultiStep', '#4d6cfa'], 
    '2':["2", '#de0d92'], # 'mstep_value_latent':["MultiStep Value-based Latents", '#de0d92'], 
    'mstep_itervaml': ["MultiStep iterVAML", '#873cd6'],
    '3': ['3', '#0a5c36']
}

## modify
BATCH_SIZE = 8
CUR_PLOTS = {'1': '/Users/leeso/OneDrive - University of Toronto/Desktop/CSC/CSC4/project/stats/1', 
             '2': "/Users/leeso/OneDrive - University of Toronto/Desktop/CSC/CSC4/project/stats/2", 
             '3': "/Users/leeso/OneDrive - University of Toronto/Desktop/CSC/CSC4/project/stats/3",
             'base': "/Users/leeso/OneDrive - University of Toronto/Desktop/CSC/CSC4/project/stats/base_16",
             'itervaml': '/Users/leeso/OneDrive - University of Toronto/Desktop/CSC/CSC4/project/stats/itervaml_16/crafter_itervaml_r', 
             'mstep_itervaml': "/Users/leeso/OneDrive - University of Toronto/Desktop/CSC/CSC4/project/stats/mstep_itervaml_8/crafter_itervaml_r", 
             'mstep': "/Users/leeso/OneDrive - University of Toronto/Desktop/CSC/CSC4/project/stats/mstep_8"}

## keep
CUR_LIST = list(CUR_PLOTS.keys())
print(CUR_LIST)
PLOT_DIR = f'../../plots{BATCH_SIZE}'
outdir = f'../../score{BATCH_SIZE}'
inpaths = [f'{outdir}/{x}.json' for x in CUR_LIST]
legend = {x:MAPPINGS[x][0] for x in CUR_LIST}
colors = [MAPPINGS[x][1] for x in CUR_LIST]


if __name__ == "__main__":
    # generate stats
    for x in CUR_PLOTS.keys(): read_stats(indir=CUR_PLOTS[x], outdir=outdir, task='', method=x)
    # import json 
    # import pathlib
    # for i in inpaths: 
    #     print(i)
    #     loaded = json.loads(pathlib.Path(i).read_text())
    
    # plot
    for i in range(len(CUR_LIST)): plot_counts(inpaths[i], f'{PLOT_DIR}/{CUR_LIST[i]}_counts.pdf', colors[i])
    plot_reward(inpaths, f'{PLOT_DIR}/reward.pdf', legend, colors)
    plot_scores(inpaths, f'{PLOT_DIR}/scores.pdf', legend, colors, ylim=12)
    plot_spectrum(inpaths, f'{PLOT_DIR}/spectrum-reward.pdf', legend, colors)


