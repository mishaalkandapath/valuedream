import torch 
import torch.nn as nn

from collections import OrderedDict
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
            depth = 2 ** i * self.cnn_depth
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
    pass