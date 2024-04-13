import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
import tensorflow_probability as tfp
import numpy as np

import common
import expl

class Agent(common.Module):

  def __init__(self, config, obs_space, act_space, step):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.tfstep = tf.Variable(int(self.step), tf.int64)
    self.wm = WorldModel(config, obs_space, self.tfstep)
    self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)
    if config.expl_behavior == 'greedy':
      self._expl_behavior = self._task_behavior
    else:
      self._expl_behavior = getattr(expl, config.expl_behavior)(
          self.config, self.act_space, self.wm, self.tfstep,
          lambda seq: self.wm.heads['reward'](seq['feat']).mode())

  @tf.function
  def policy(self, obs, state=None, mode='train'):
    """
    Given an observation and prev_state (a tuple w the prev stochastic state and prev action), embed the observation.
    Then use the embedding, action, and stochastic state to get the next stochastic state+deterministic state representation. 
    Use these features to sample the next action. 
    After you get 
    return the action and the new state
    """
    # print("---At Policy--")
    # print("Is state none? {}".format(state is None))
    # print("Obs shape we have is: {}".format(obs['image'].shape))
    obs = tf.nest.map_structure(tf.tensor, obs)
    tf.py_function(lambda: self.tfstep.assign(
        int(self.step), read_value=False), [], [])
    if state is None:
      latent = self.wm.rssm.initial(len(obs['reward']))
      action = tf.zeros((len(obs['reward']),) + self.act_space.shape) # generate the zero action giving shape batchxaction_space: (b, 17)
      state = latent, action
    latent, action = state
    embed = self.wm.encoder(self.wm.preprocess(obs)) #preprocessing just make a reward clipping function, and initializes the discount factors with 0 for terminal states
    sample = (mode == 'train') or not self.config.eval_state_mean
    latent, _ = self.wm.rssm.obs_step(
        latent, action, embed, obs['is_first'], sample) # get the posteriors
    feat = self.wm.rssm.get_feat(latent) # a concatenation of the stochastic and determinisitc state
    if mode == 'eval':
      actor = self._task_behavior.actor(feat)
      action = actor.mode()
      noise = self.config.eval_noise
    elif mode == 'explore':
      actor = self._expl_behavior.actor(feat)
      action = actor.sample()
      noise = self.config.expl_noise
    elif mode == 'train':
      actor = self._task_behavior.actor(feat)
      action = actor.sample()
      noise = self.config.expl_noise
    action = common.action_noise(action, noise, self.act_space)
    outputs = {'action': action}
    state = (latent, action)
    return outputs, state

  @tf.function
  def train(self, data, state=None):
    # tf.print("Graph execution")
    metrics = {}
    state, outputs, mets = self.wm.train(data, state) #one- step prediction given the current observatiion and states
    metrics.update(mets)
    start = outputs['post']
    reward = lambda seq: self.wm.heads['reward'](seq['feat']).mode()
    metrics.update(self._task_behavior.train(
        self.wm, start, data['is_terminal'], reward)) # train the actor-critic model
    if self.config.expl_behavior != 'greedy':
      mets = self._expl_behavior.train(start, outputs, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    # print("DONE TRAINING A2C")
    # print("METRICS IN TRAIN FXN", metrics)
    return state, metrics

  @tf.function
  def report(self, data):
    report = {}
    data = self.wm.preprocess(data)
    for key in self.wm.heads['decoder'].cnn_keys:
      name = key.replace('/', '_')
      report[f'openl_{name}'] = self.wm.video_pred(data, key)
    return report


class WorldModel(common.Module):

  def __init__(self, config, obs_space, tfstep):
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    self._changed = False
    self.config = config
    self.tfstep = tfstep
    self.rssm = common.EnsembleRSSM(**config.rssm)
    self.encoder = common.Encoder(shapes, **config.encoder)
    self.heads = {}
    self.heads['decoder'] = common.RecurrentDecoder(shapes, **config.decoder) if self._changed else common.Decoder(shapes, **config.decoder)
    self.heads['reward'] = common.MLP([], **config.reward_head)
    if config.pred_discount:
      self.heads['discount'] = common.MLP([], **config.discount_head)
    for name in config.grad_heads:
      assert name in self.heads, name
    self.model_opt = common.Optimizer('model', **config.model_opt)
    self.reg_measures = {"auto_corr":[], "norm":[]}
    self.post_feat = None

  def train(self, data, state=None):
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    # print("DONE INFERENCE PART")
    modules = [self.encoder, self.rssm, *self.heads.values()]
    metrics.update(self.model_opt(model_tape, model_loss, modules))
    # print("DONE UPDATE")
    # print(metrics)
    return state, outputs, metrics
    

  def multi_step_helper(self, data):
    #convert the matrix into what we want:
    # just swap the first two indices
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    images = data["image"]
    images = swap(images)
    # concatenate to get the following: the entire image sequence, 
    #             the entire image sequence except for the first, the entire image sequence except for first two...
    new_images = tf.concat([images[i:, :] for i in range(0, 5)], 0)
    return swap(new_images)    

  def loss(self, data, state=None):
    # print("At Loss, data shape: {} {}".format(data["image"].shape, data["action"].shape))
    # with tf.Session() as sess: print("the first few actions are {}".format(data["action"][0, 0:3].eval())) 
    data = self.preprocess(data)
    embed = self.encoder(data)
    # print("PROCESSED WITH ENCODER")
    # print("state is none? {}".format(state is None))
    # print("but we have data of shape {}".format(data["image"].shape))  
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
    # print("DONE OBSERVING")
    kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl) # kl loss between post and prior
    assert len(kl_loss.shape) == 0
    likes = {}
    losses = {'kl': kl_loss}
    feat = self.rssm.get_feat(post)
    self.post_feat = tf.stop_gradient(feat)
    avgnorm = tf.reduce_mean(tf.norm(tf.stop_gradient(tf.identity(feat)), axis=2))
    
    for name, head in self.heads.items():
      # print("heyy ", name)
      grad_head = (name in self.config.grad_heads)
      inp = feat if grad_head else tf.stop_gradient(feat)
      out = head(inp, data["action"]) if name == "decoder" and self._changed else head(inp)
      # print("DONE HEAD {}".format(name))
      dists = out if isinstance(out, dict) else {name: out}
      # print("for head {} we have {} ".format(name, list(dists.keys())))
      for key, dist in dists.items(): #loss on the log probability of the true vakue being observed under the predicted distribution for all the heads
        if name == "decoder" and self._changed:
          like = tf.cast(dist.log_prob(self.multi_step_helper(data)), tf.float32)
        else: like = tf.cast(dist.log_prob(data[key]), tf.float32)
        likes[key] = like
        losses[key] = -like.mean()
      # print("DONE LOSS {}".format(name))
    model_loss = sum(
        self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, kl=kl_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics['model_kl'] = kl_value.mean()
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    metrics["avg_norm"] = avgnorm
    last_state = {k: v[:, -1] for k, v in post.items()}   # NOTE: why is this the last state? print the shape of the v
    return model_loss, last_state, outs, metrics

  def imagine(self, policy, start, is_terminal, horizon): # start is basically the posterior state, it has its logits, stoch and det vectors
    # print("---At Imagine--")
    # print("In start: {}".format(list(start[key].shape for key in start))) 
    #Initially start vectors are of shape 16, 50, 32, 32
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:])) # flatten the time dimensions
    start = {k: flatten(v) for k, v in start.items()} # now of shape 800, 32, 32
    start['feat'] = self.rssm.get_feat(start)
    start['action'] = tf.zeros_like(policy(start['feat']).mode())
    seq = {k: [v] for k, v in start.items()}
    for _ in range(horizon):
      action = policy(tf.stop_gradient(seq['feat'][-1])).sample()
      state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action) # get a prior from the last state in the sequence so far and the action
      feat = self.rssm.get_feat(state)
      for key, value in {**state, 'action': action, 'feat': feat}.items():
        seq[key].append(value) # grow the sequence
    seq = {k: tf.stack(v, 0) for k, v in seq.items()}
    if 'discount' in self.heads:
      disc = self.heads['discount'](seq['feat']).mean()
      if is_terminal is not None:
        # Override discount prediction for the first step with the true
        # discount factor from the replay buffer.
        true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
        true_first *= self.config.discount
        disc = tf.concat([true_first[None], disc[1:]], 0)
    else:
      disc = self.config.discount * tf.ones(seq['feat'].shape[:-1])
    seq['discount'] = disc
    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.
    seq['weight'] = tf.math.cumprod(
        tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0)
    return seq

  @tf.function
  def preprocess(self, obs):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_'):
        continue
      if value.dtype == tf.int32:
        value = value.astype(dtype)
      if value.dtype == tf.uint8:
        value = value.astype(dtype) / 255.0 - 0.5
      obs[key] = value
    obs['reward'] = {
        'identity': tf.identity,
        'sign': tf.sign,
        'tanh': tf.tanh,
    }[self.config.clip_rewards](obs['reward'])
    obs['discount'] = 1.0 - obs['is_terminal'].astype(dtype)
    obs['discount'] *= self.config.discount
    return obs

  @tf.function
  def video_pred(self, data, key):
    decoder = self.heads['decoder']
    truth = data[key][:6] + 0.5
    embed = self.encoder(data)
    states, _ = self.rssm.observe(
        embed[:6, :5], data['action'][:6, :5], data['is_first'][:6, :5])
    recon = decoder(self.rssm.get_feat(states))[key].mode()[:6] if not self._changed else decoder(self.rssm.get_feat(states), data['action'][:6, :5], hor=1)[key].mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:6, 5:], init)
    openl = decoder(self.rssm.get_feat(prior))[key].mode() if not self._changed else decoder(self.rssm.get_feat(prior), data['action'][:6, 5:], hor=1)[key].mode()
    model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    video = tf.concat([truth, model, error], 2)
    B, T, H, W, C = video.shape
    return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


class ActorCritic(common.Module):

  def __init__(self, config, act_space, tfstep):
    self.config = config
    self.act_space = act_space
    self.tfstep = tfstep
    discrete = hasattr(act_space, 'n')
    if self.config.actor.dist == 'auto':
      self.config = self.config.update({
          'actor.dist': 'onehot' if discrete else 'trunc_normal'})
    if self.config.actor_grad == 'auto':
      self.config = self.config.update({
          'actor_grad': 'reinforce' if discrete else 'dynamics'})
    self.actor = common.MLP(act_space.shape[0], **self.config.actor)
    self.critic = common.MLP([], **self.config.critic)
    if self.config.slow_target:
      self._target_critic = common.MLP([], **self.config.critic)
      self._updates = tf.Variable(0, tf.int64)
    else:
      self._target_critic = self.critic
    self.actor_opt = common.Optimizer('actor', **self.config.actor_opt)
    self.critic_opt = common.Optimizer('critic', **self.config.critic_opt)
    self.rewnorm = common.StreamNorm(**self.config.reward_norm)

  def train(self, world_model, start, is_terminal, reward_fn):
    metrics = {}
    hor = self.config.imag_horizon
    # The weights are is_terminal flags for the imagination start states.
    # Technically, they should multiply the losses from the second trajectory
    # step onwards, which is the first imagined step. However, we are not
    # training the action that led into the first step anyway, so we can use
    # them to scale the whole sequence.
    with tf.GradientTape() as actor_tape:
      print("START SHAPE::",start)
      seq = world_model.imagine(self.actor, start, is_terminal, hor) # generate an imagined trajectory upto horizon H 
      reward = reward_fn(seq)
      seq['reward'], mets1 = self.rewnorm(reward)
      mets1 = {f'reward_{k}': v for k, v in mets1.items()}
      target, mets2 = self.target(seq)
      print("TARGET::", target)
      # TARGET:: Tensor("stack_5:0", shape=(15, 400), dtype=float32)
      actor_loss, mets3 = self.actor_loss(seq, target) # train the actor here based on teh value predictions from the target
    with tf.GradientTape() as critic_tape:
      # critic_loss, mets4 = self.critic_loss(seq, target) # train the critic
      critic_loss, mets4 = self.critic_itervaml(seq, world_model.post_feat)
    metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
    metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
    metrics.update(**mets1, **mets2, **mets3, **mets4)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, seq, target):
    # Actions:      0   [a1]  [a2]   a3
    #                  ^  |  ^  |  ^  |
    #                 /   v /   v /   v
    # States:     [z0]->[z1]-> z2 -> z3
    # Targets:     t0   [t1]  [t2]
    # Baselines:  [v0]  [v1]   v2    v3
    # Entropies:        [e1]  [e2]
    # Weights:    [ 1]  [w1]   w2    w3
    # Loss:              l1    l2
    metrics = {}
    # Two states are lost at the end of the trajectory, one for the boostrap
    # value prediction and one because the corresponding action does not lead
    # anywhere anymore. One target is lost at the start of the trajectory
    # because the initial state comes from the replay buffer.
    policy = self.actor(tf.stop_gradient(seq['feat'][:-2]))
    if self.config.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.config.actor_grad == 'reinforce':
      baseline = self._target_critic(seq['feat'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      action = tf.stop_gradient(seq['action'][1:-1])
      objective = policy.log_prob(action) * advantage
    elif self.config.actor_grad == 'both':
      baseline = self._target_critic(seq['feat'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(seq['action'][1:-1]) * advantage
      mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.config.actor_grad)
    ent = policy.entropy()
    ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = tf.stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    # States:     [z0]  [z1]  [z2]   z3
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]   v3
    # Weights:    [ 1]  [w1]  [w2]   w3
    # Targets:    [t0]  [t1]  [t2]
    # Loss:        l0    l1    l2
    dist = self.critic(seq['feat'][:-1])
    target = tf.stop_gradient(target)
    weight = tf.stop_gradient(seq['weight'])
    critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
    metrics = {'critic': dist.mode().mean()}
    return critic_loss, metrics

  def critic_itervaml(self, seq, code_vecs):
    # first reshape seq["feat"][:-1] to be a vector
    restructured_seq = self.reshape_seq(seq["feat"][:-1], code_vecs.shape[1], code_vecs.shape[0])
    
    # call the critic on it to get distribution
    
    # next reshape code_vecs to be a vector, call dist on it, get mean
    
    
    # -log_prob.mean()
    pass

  def reshape_seq(self, seq, obslen, n_batches):
    print(seq, obslen, n_batches)
    hor = self.config.imag_horizon
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    flat_seq = flatten(seq)
    extra_mask=[]  # hor*(hor-1)
    for i in range(1,hor): extra_mask.extend([True]*(hor-i)+[False]*i)
    batch_mask = tf.constant([True]*hor*(obslen-hor+1)+extra_mask)
    mask = tf.concat([batch_mask] * n_batches, axis=0)
    # print("FLATTENED MASK", mask)
    # print("FLATTENED SEQ", flat_seq)
    print("REMOVED", tf.boolean_mask(flat_seq, tf.reshape(mask, (-1, 1))))
        
    # first: remove unneeded
    
    # row = seq[:][0][:]
    # print("ROWWWW", row)
    
    # reshape_batch = lambda x: tf.concat([x[i:i+hor] if i <= (obslen - hor) else x[i:] for i in range(obslen)]) # row/batch = (50,) -> (50,15)
    # all_batches = tf.concat([reshape_batch(seq[i*obslen:(i+1)*obslen]) for i in range(n_batches)], 0)
    # return all_batches
  
  def critic_itervaml_attempt1(self, seq, code_vecs):
    # States:     [z0]  [z1]  [z2]   z3
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]   v3
    # Weights:    [ 1]  [w1]  [w2]   w3
    # Targets:    [t0]  [t1]  [t2]
    # Loss:        l0    l1    l2
        
    dist = self.critic(seq['feat'][:-1]) # TODO: why is it -1? what does that mean
    # NOTE: should i stop gradients on the posterior? itervaml does not seem to 
    print("ESTIMATED VALUE::", dist)
    # tfp.distributions.Independent("IndependentNormal_5", batch_shape=[15, 400], event_shape=[], dtype=float32)
    print(seq['feat'][:-1]) 
    # Tensor("strided_slice_74:0", shape=(15, 400, 2048), dtype=float32)
    print("CODE VECS ", code_vecs)
    # code_vecs = code_vecs.reshape([-1] + list(code_vecs.shape[2:])) # flatten the time dimensions
    # print("CODE FLATTENED:", code_vecs)
    # CODE VECS  Tensor("concat:0", shape=(8, 50, 2048), dtype=float32)
    # TODO: now translate the code_vecs into value predictions self.critic(code_vecs), compare this shape to target
    estimated_code_value = self.critic(code_vecs).mean()  # NOTE: using expected value from post val dist, but is KL better? 
    print("ESTIMATED VALUE OF CODE VECS ", estimated_code_value)
    # tfp.distributions.Independent("IndependentNormal_6", batch_shape=[8, 50], event_shape=[], dtype=float32)
    reshaped_batch, mask = self.itervaml_helper(estimated_code_value)
    neg_loglike = -(dist.log_prob(reshaped_batch))
    
    masked = tf.where(mask, 0.0, neg_loglike)
    tf.debugging.check_numerics(masked, "post masking")
    print("NEGLOGLIKE::", neg_loglike, masked)
    critic_loss = masked.mean()
    metrics = {'critic': dist.mode().mean()}
    return critic_loss, metrics

  def itervaml_helper(self, post_val):
    # NOTE: this code won't work well if config.imag_horizon >> seqlen
    # post_val shape = (batch, seqlen) == (16,50)
    # output: (horizon, batch*seqlen) == (15,800)
    hor = self.config.imag_horizon
    seqlen = post_val.shape[1]
    reshape_batch = lambda x: tf.stack([x[i:i+hor] if i <= (seqlen - hor) 
                                        else tf.pad(x[i:], tf.constant([[0,hor-(seqlen-i)]]), "CONSTANT", constant_values=float('nan')) for i in range(seqlen)]) # row/batch = (50,) -> (50,15)
    all_batches = tf.transpose(tf.concat([reshape_batch(post_val[i]) for i in range(post_val.shape[0])], 0), [1,0])
    mask = tf.math.is_nan(all_batches)
    return tf.where(mask, 0.0, all_batches), mask

  def target(self, seq):
    # States:     [z0]  [z1]  [z2]  [z3]
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]  [v3]
    # Discount:   [d0]  [d1]  [d2]   d3
    # Targets:     t0    t1    t2
    reward = tf.cast(seq['reward'], tf.float32)
    disc = tf.cast(seq['discount'], tf.float32)
    value = self._target_critic(seq['feat']).mode()
    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.config.discount_lambda,
        axis=0)
    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, metrics

  def update_slow_target(self):
    if self.config.slow_target:
      if self._updates % self.config.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.config.slow_target_fraction)
        for s, d in zip(self.critic.variables, self._target_critic.variables):
          d.assign(mix * s + (1 - mix) * d)
      self._updates.assign_add(1)

# def calc_autocorrelation(code_vecs):
#     return tf.reduce_mean(tf.norm(code_vecs, axis=2))
#     # print("DID I GET THERE?")
#     # # code vecs in 16, 50, len code vecs
#     # autocorrelation = tfp.stats.auto_correlation(code_vecs, max_lags=(code_vecs.shape[1]-1), axis=1)
#     # # autocorrelation = autocorrelation[:,1:,:]
#     # # avg_corr = tf.reduce_mean(tf.reduce_sum(autocorrelation, axis=2))
#     # avg_corr = tf.math.count_nonzero(tf.math.is_nan(autocorrelation))
    
#     # # norm avgs
#     # avg_code_norm = tf.reduce_mean(tf.norm(code_vecs, axis=2))
#     # # with tf.Session() as sess:
#     # # with tf.compat.v1.Session() as sess:
#     # # # print(avg_corr.numpy())    
#     # # # self.reg_measures["auto_corr"].append(tf.keras.backend.eval(avg_corr))
#     # #   print(sess.run(avg_corr))
#     # #   # self.reg_measures["norm"].append(avg_corr.eval())
#     # #   # self.reg_measures["norm"].append(avg_code_norm.eval())
    
#     # # print("BEGINNING IO???")
#     # # if len(self.reg_measures) % 1 == 0: 
#     # #   pd.DataFrame(self.reg_measures).to_csv("measures.csv", index=False)
#     # print("COMPLETED IO???????")
#     # return avg_corr, avg_code_norm