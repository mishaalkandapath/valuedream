import torch 
import torch.nn as nn

from collections import OrderedDict
import utils
import optree
"""
The world model consists of an image encoder, a Recurrent State-Space Model
(RSSM; Hafner et al., 2018) to learn the dynamics, and predictors for the image, reward, and discount
factor. The world model is summarized in Figure 2. The RSSM uses a sequence of deterministic
recurrent states ht, from which it computes two distributions over stochastic states at each step. The
posterior state zt incorporates information about the current image xt, while the prior state ˆzt aims
to predict the posterior without access to the current image. 

The concatenation of deterministic and stochastic states forms the compact model state. From the posterior model state, we reconstruct the
current image xt and predict the reward rt and discount factor γt.

The model state is the concatenation of deterministic GRU state and a sample of the stochastic state. The transition, reward, and discount predictors are MLPs
"""

#first we have the encoder that given an image observation will compute the stoachstic state z that will be target of the transition and the representation model
# note the transition model is called the prior and the representation model is called the posterior

class Encoder(nn.Module):
    """
    @shapes: information about the shapes in the obsercation space. shapes["image"] gives the Box object encasing the shape of the observation space. 
    Similarly there are is_last, is_first, is_terminal, and reward boxes, each have their own 'shape' attr coz they are gym.spaces.Box objects
    """
    def __init__(self, shapes, cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400], batch_norm=False, act="elu"):
        self.og_channels = 3 
        self.obs_space = shapes["image"]
        self.cnn = self.make_cnn()
        self.act = act
        self.cnn_depth = cnn_depth
        self.cnn_kernels = cnn_kernels
        self.batch_norm = batch_norm

    def make_cnn(self):
        o_d = OrderedDict()
        for i, kernel in enumerate(self.cnn_kernels):
            depth = 2 ** i * self.cnn_depth #no of channels in the output
            if i == 0: in_channels = self.og_channels #channel is at the end here. 
            cnn = nn.Conv2d(in_channels, depth, kernel, stride=2) #no of channgels in the input, no of channels in the output, kernel size
            o_d[f'conv{i}'] = cnn
            #make a batch_norm: in the original code they use layer norm. idk how to use layer norm for cnn so i use batch norm here lol
            if self.batch_norm: 
                bn_layer = nn.LayerNorm()
                o_d[f'bn{i}'] = bn_layer
            o_d[f'{self.act}{i}'] = nn.ELU()   
            in_channels = depth
        return nn.Sequential(o_d)

    def make_mlp(self):
        pass # we don';t really need this for our purposes

    def forward(self, obs_image):
        # time to encode the observation image
        x = obs_image.reshape((-1,) + tuple(obs_image.shape[-3:]))
        x = x.permute(0, 3, 1, 2) # change the channel to the front
        x = self.cnn(obs_image)
        return self.cnn(obs_image).reshape(obs_image.shape[0], -1) #flatten the image to a vector

class Decoder(nn.Module):
    def __init__(self, in_size, out_shape=(3, 64, 64), cnn_depth=48, cnn_kernels=(4, 4, 4, 4), batch_norm=False, act="elu"):
        self.in_size = in_size
        self.out_shape = out_shape 
        self.act = act
        self.cnn_depth = cnn_depth
        self.cnn_kernels = cnn_kernels
        self.batch_norm = batch_norm
    
    def make_cnn(self):
        o_d = OrderedDict()
        o_d["fc1"] = nn.Linear(self.in_size, 32 * self.cnn_depth)
        for i, kernel in enumerate(self.cnn_kernels):
            depth = 2 ** (len(self.cnn_kernels) -i - 2) * self.cnn_depth 

            act, norm = self.act, self.batch_norm
            if i == len(self.cnn_kernels) - 1:
                depth, act, norm = 3, nn.Identity(), False
            if i == 0: in_channels = 32 * self.cnn_depth
            cnn = nn.ConvTranspose2d(in_channels, depth, kernel, stride=2)
            o_d[f'conv{i}'] = cnn
            if norm: 
                bn_layer = nn.LayerNorm()
                o_d[f'bn{i}'] = bn_layer
            if act is not None: o_d[f'{act}{i}'] = nn.ELU()
            in_channels = depth
        return nn.Sequential(o_d)   

    def forward(self, x):
        out = self.cnn(x)
        mean = out.reshape(x.shape[:-1] + self.out_shape)
        mean = mean.permute(0, 1, 3, 4, 2)
        return torch.distributions.independent.Independent(torch.distributions.Normal(mean, 1.0), 3) #idk what this is 
    
#big boi
class RSSM(nn.Module):
    def __init__(self, encoder_out_shape, ensemble=5, stoch=30, deter=200, hidden=200, discrete=32, act="elu", norm=False, std_act="softplus", min_std=0.1):
        self.ensemble = ensemble
        self.stoch = stoch # stochastic state dimensions
        self.deter = deter # deterministic state dimensions
        self.hidden = hidden #GRU hidden state
        self.discrete = discrete #discretize representation
        self.act = nn.ReLU() if act == "relu" else nn.ELU()
        self.norm = nn.LayerNorm() if norm else nn.Identity()
        self.std_act = nn.ReLU() if std_act == "relu" else nn.Softplus()
        self.min_std = min_std

        self.gru_cell = GRUCell(self.hidden, self.deter)

        #layers called elsewhere -- these are more or less just normal non-linearities in the middle of the network
        self.img_in = nn.Linear(self.stoch*self.discrete+16, self.hidden)

        for i in range(self.ensemble):
            setattr(self, f"img_out_{i}", nn.Linear(self.stoch+self.discrete+16))  #because there are 16 possible actions in crafter
            setattr(self, f"img_out_norm{i}", self.norm)
            setattr(self, f"img_dist_{i}",  nn.Linear(self.hidden, self.discrete*self.stoch))

        self.obs_out = nn.Linear(self.deter + encoder_out_shape, self.hidden)
        self.obs_dist = nn.Linear(self.hidden, self.stoch*self.discrete)

    def initial(self, batch_size): # initialize the state of the RSSM - the initial state of the RSSM:
        deter = torch.zeros(batch_size, self.deter)
        logit_state = torch.zeros(batch_size, self.stoch, self.discrete) #
        stochastic_state = torch.zeros(batch_size, self.stoch, self.discrete)
        return logit_state, stochastic_state, deter

    def observe(self, embedding, action, is_first, state=None): #this is the recurrent model part of the RSSM, plus the representation and transition predictors
        #embeddings - zt, action: at, state: ht
        swap = lambda x: torch.transpose(x, 1, 0)

        if state is None: state = self.initial(embedding.shape[0]) #means we be starting from scratch state
        post, prior = utils.static_scan(lambda prev, inputs: self.obs_step(prev[0], *inputs), #calle obs_step to get the posteriors and the priors, where prev is the prev)_state
                                        (swap(action), swap(embedding), swap(is_first)), (state, state))   #here we generate the post and prior 
        for i in len(post):
            post[i] = swap(post)
            prior[i] = swap(prior)

        return post, prior
    
    def imagine(self, action, state): 
        swap = lambda x: torch.transpose(x, 1, 0)

        if state is None: state = self.initial(action.shape[0])
        
        action = swap(action)
        prior = utils.static_scan(self.img_step, action, state)
        for i in len(prior):
            prior[i] = swap(prior[i])
        return prior
    
    def get_feat(self, state): #i dont know what this function is for yet -- used in the agent stuff
        _, stoch, deter = state
        shape = stoch.shape[:-2] + [self.stoch * self.discrete]
        stoch = torch.reshape(stoch, shape)
        return torch.concat([stoch, deter], -1)

    
    def obs_step(self, prev_state, prev_action, embedding, is_first, sample=True): #this is posterior since it gets the image embedding
        take_everthing_except_first = lambda x: (torch.einsum('b,b...->b...', 1.0 - is_first.astype(x.dtype), x))
        prev_state, prev_action = optree.tree_map(take_everthing_except_first, [prev_state, prev_action])

        _, prior_deter, _ = self.img_step(prev_state, prev_action, sample)

        x = torch.cat([prior_deter, embedding], -1)
        x = self.obs_out(x)
        x = self.norm(x)
        x = self.act(x)

        x = self.obs_dist(x)
        x = x.reshape(list(x.shape[:-1]) + [self.stoch, self.discrete])
        dist = torch.distributions.independent.Independent(torch.distributions.one_hot_categorical.OneHotCategorical(logits=x.astype(torch.float32)), 1)
        stoch = dist.sample() if sample else dist.mode()

        return (stoch, prior_deter, x)

    def img_step(self, prev_state, prev_action, sample=True): #this is the prior since it has no embedding
        """
        Given previous stochastic state and previous action, get the next stochastic state 
        (plus the current deterministic state in the cell, plus the emnsemble of stats generated in between)
        """
        _, prev_stoch, deter = prev_state #take the stochastic state out
        
        #collapse the last two to direct concat
        shape = list(prev_stoch.shape[:-2]) + [self.stoch + self.discrete]
        prev_stoch = prev_stoch.reshape(shape)
        
        x = torch.cat([prev_stoch, prev_action], -1)
        x = self.img_in(x)
        if self.norm: x = self.norm(x)
        x = self.act(x)

        x, deter = self.gru_cell(x, [deter])
        deter = deter[0]
        stats = self.suff_stats_ensemble(x)

        index = torch.randint(0, self.ensemble, ())
        x = stats[index]
        dist = torch.distribution.independent.Independent(torch.distributions.one_hot_categorical.OneHotCategorical(logits=x.astype(torch.float32)), 1) #might wanna do implement straight through gradients here 
        stoch = dist.sample() if sample else dist.mode()
        return (stoch, deter, stats)

    def suff_stats_ensemble(self, x):
        bs = list(x.shape[:-1])
        inp = x.reshape([-1, x.shape[-1]])
        stats = []

        for i in range(self.ensemble):
            x = getattr(self, f"img_out_{i}")(inp)
            x = getattr(self, f"img_out_norm{i}")(x)
            x = self.act(x)
            x = getattr(self, f"img_dist_{i}")(x)
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            stats.append(logit)

        stats = torch.stack([x for x in stats], 0) # stack logits
        stats = stats.reshape([stats.shape[0]] + bs + list(stats.shape[2:]))
        return stats
    
    def kl_loss(self, post, prior, forward, balance, free, free_avg):
        dist = torch.distributions.one_hot_categorical.OneHotCategorical
        kld = torch.distributions.kl.kl_divergence
        lhs, rhs = (prior[2], post[2]) if forward else (post[2], prior[2]) # 2 is where the logits are at
        mix = balance if forward else (1-balance)

        if balance == 0.5: #no balancing, so j do it once and prop grads through
            value = kld(dist(logits=lhs), dist(logits=rhs))
            loss = torch.maximum(value, free).mean()
        else:
            #KL Balancing:
            value_lhs = value = kld(dist(lhs), dist(rhs.detach()))
            value_rhs = kld(dist(lhs.detach()), dist(rhs))

            if free_avg:
                loss_lhs = torch.maximum(value_lhs.mean(), free) # free is 1, so its saying at most 1. 
                loss_rhs = torch.maximum(value_rhs.mean(), free)
            else:
                loss_lhs = torch.maximum(value_lhs, free).mean()
                loss_rhs = torch.maximum(value_rhs, free).mean()
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
            return loss, value



#directly pasted from repository: CREDIT REPO HERE
class GRUCell(nn.Module):

  def __init__(self, inp_size,
               size, norm=False, act=torch.tanh, update_bias=-1):
    super(GRUCell, self).__init__()
    self._inp_size = inp_size
    self._size = size
    self._act = act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = nn.Linear(inp_size+size, 3*size,
                            bias=norm is not None)
    if norm:
      self._norm = nn.LayerNorm(3*size)

  @property
  def state_size(self):
    return self._size

  def forward(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(torch.cat([inputs, state], -1))
    if self._norm:
      parts = self._norm(parts)
    reset, cand, update = torch.split(parts, [self._size]*3, -1)
    reset = torch.sigmoid(reset)
    cand = self._act(reset * cand)
    update = torch.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]