import crafter, gym, numpy as np, matplotlib.pyplot as plt
def generate_random_sequence(num_steps = 4):
    """
    Function to just start crafter game and make a sequence
    Need int16 because we use subtraction at one point!
    """
    env = gym.make('CrafterReward-v1')  # Or CrafterNoReward-v1
    env = crafter.Recorder(
    env, './path/to/logdir',
    save_stats=True,
    save_video=False,
    save_episode=False,
    )

    obs = env.reset()
    seq = []
    actions = [3, 3, 3, 3]
    for step in range(num_steps):
        action = actions[step] # np.random.randint(1, 5)
        obs, reward, done, info = env.step(action)
        seq.append((obs.astype(np.int16), int(action), reward))
    return seq

def get_intersection_from_sequence(sequence: list, fill_fn = None):
    """
    seq: list of tuples (obs, a, r) where action a, yielded observation obs & reward r
    returns one obs that compresses the sequence
    """
    # crop to just the map
    seq = [(o[:49, :-1, :], a, r) for o,a,r in sequence]
    curr_intersection = seq[0][0]
    # iteratively intersect
    l = [(seq[0][0], 'o1')]
    for i in range(len(seq)):
        if i == 0: continue  # skip first
        obs, a, r = seq[i]
        curr_intersection = intersect(curr_intersection, obs, a)
        curr_intersection = curr_intersection.astype(np.int16)
        l.append((curr_intersection, f'intersection {i+1}'))
    l += [(x[0], f'obs{i}') for i, x in enumerate(seq)]
    show(l)
        
    # TODO now fill in the regions that were lost, since some moves can "undo" cropped regions we lost we need to 
    # see which ones were actually still lost
    # lost_regions = find_lost_regions(curr_intersection, lost_regions)
    # filled_intersection = one_obs_fill(lost_regions, curr_intersection, seq[0][0])
    
    # Fill back in the toolbar and righthand column
    original = sequence[0][0].astype(np.int16)
    result = np.concatenate((
            np.concatenate(
                (
                    curr_intersection, 
                    original[49:, :-1, :]
                ), 
                axis=0
            ),
            original[:, -1:, :]
        ), 
        axis=1
    )
    show([(result, 'final')], ncols=1)

def one_obs_fill(lost_regions, obs, ref_obs):
    """
    Just fill all lost regions of obs with the content from one observation ref_obs
    """
    nobs = np.copy(obs)
    for l in lost_regions:
        if l is not None:
            r, c, ch = l
            rl, rr = r
            cl, cr = c
            chl, chr = ch
            nobs[rl:rr, cl:cr, chl:chr] = ref_obs[rl:rr, cl:cr, chl:chr]
    return nobs

def show(matrices, ncols=4):
    """
    Nice little helper to show plots 
    """
    num_matrices = len(matrices)
    nrows = (num_matrices + ncols - 1) // ncols  # Calculate number of rows based on ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))

    for i, (matrix, label) in enumerate(matrices):
        row = i // ncols
        col = i % ncols

        if nrows == 1:  # Handle single row case
            ax = axes[col]
        else:
            ax = axes[row, col]

        ax.imshow(matrix, cmap='viridis', interpolation='nearest')
        ax.axis('off')  # Hide the axis
        ax.set_title(label, fontsize=12)
        ax.set_aspect('equal')  # Set aspect ratio to be equal

    # Adjust layout to prevent overlap
    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    plt.show()

def intersect(obs1, obs2, action):
    """
    obs1, obs2: both numpy matrices of the same shape
    action is the action that took obs1 to obs2
    returns the intersection
    """
    shift = 7
    thresh = 4      # how much r/g/b pixel can differ by to be different
    rows, cols, channels = obs1.shape[0], obs1.shape[1], obs1.shape[2]
    if action == 1:
        lost_shape = (rows, shift, channels)
        cropped_obs = np.concatenate((np.zeros(lost_shape), obs1[:, :-shift, :]), axis=1)
        return np.where(abs(cropped_obs - obs2) < thresh, obs2, 0)
    elif action == 2:
        lost_shape = (rows, shift, channels)
        cropped_obs = np.concatenate((obs1[:, shift:, :], np.zeros(lost_shape)), axis=1)
        return np.where(abs(cropped_obs - obs2) < thresh, obs2, 0)
    elif action == 3:
        lost_shape = (shift, cols, channels)
        cropped_obs = np.concatenate((np.zeros(lost_shape), obs1[:-shift, :, :]), axis=0)
        return np.where(abs(cropped_obs - obs2) < thresh, obs2, 0)
    elif action == 4:
        lost_shape = (shift, cols, channels)
        cropped_obs = np.concatenate((obs1[shift:, :, :], np.zeros(lost_shape)), axis=0)
        return np.where(abs(cropped_obs - obs2) < thresh, obs2, 0)
    else:
        return np.where(abs(obs1 - obs2) < thresh, obs1, 0)
   
def validate_lost_regions(obs, lost_regions):
    rows, cols = obs.shape[0], obs.shape[1]
    # TODO validate and adjust the bounds
    return lost_regions
    

get_intersection_from_sequence(generate_random_sequence(4))

