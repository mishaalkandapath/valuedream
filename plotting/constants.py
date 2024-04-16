MAPPINGS = {
    'base': ['DreamerV2', '#595457'], 
    'itervaml':['iterVAML', '#9e1946'], 
    'value_latent':["Value-based Latents", '#710627'], 
    'mstep': ['MultiStep', '#4d6cfa'], 
    'mstep_value_latent':["MultiStep Value-based Latents", '#de0d92'], 
    'mstep_itervaml': ["MultiStep iterVAML", '#873cd6']
}

BATCH_SIZE = 16

CUR_LIST = ['itervaml', ]

PLOT_DIR = f'plots{BATCH_SIZE}'

indir = f'score{BATCH_SIZE}'
inpaths = [f'{indir}/{x}.json' for x in CUR_LIST]
legend = {x:MAPPINGS[x][0] for x in CUR_LIST}
colors = [MAPPINGS[x][1] for x in CUR_LIST]
