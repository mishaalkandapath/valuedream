import torch
import torch.nn as nn
import torch.distributions as td
from dynamics import Encoder, RSSM, Decoder
from actorcritic import BaseMLP
import utils


class WorldModel(nn.Module):
    def __init__(self, config, obs_space, step):
        self.config = config #config dict
        self.obs_space = obs_space #observation space, 64, 64, 3 box
        self.step = step

        self.encoder = Encoder(obs_space)
        self.rssm = RSSM(utils.calc_conv_shape(obs_space.shape[1:]+obs_space.shape[0]))

        self.decoder = Decoder(self.rssm.stoch * self.rssm.discrete)
        self.reward_head = BaseMLP(self.rssm.stoch * self.rssm.discrete, 1, inter_units=400, layers=4, act="elu")
        self.discound_head = BaseMLP(self.rssm.stoch * self.rssm.discrete, 1, inter_units=400, layers=4, act="elu")

        self.modules = {"encoder": self.encoder, "rssm": self.rssm, "decoder": self.decoder, "reward_head": self.reward_head, "discount_head": self.discount_head}
    
    def train(self, data, opt, state=None):
        model_loss, state, outputs, metrics = self.loss(data, state)

        opt.zero_grad()
        model_loss.backward()
        opt.step()

        metrics["total_loss"] = model_loss
        return state, outputs, metrics
    
    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm.observe(embed, data["action"], state)
        
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, forward=False, balance=0.8, free=0.0, free_avg=True) #KL term in loss

        assert len(kl_loss.shape) == 0, "kl loss returned non-scalar val of size {}".format(kl_loss.shape)

        likes = {}
        losses = {"kl_loss": kl_loss}
        feat = self.rssm.get_feat(post)

        for module in self.modules: #the decoder, rewards, and discount
            if module in ("encoder", "rssm"): continue
            out = module(feat) # get prediction
            
            for dist in out:
                if module == "discount": dist = td.independent.Independent(td.bernoulli.Bernoulli(logits=dist), 1)
                else: dist = td.independent.Independent(td.normal.Normal(dist, 1), 3 if module == "decoder" else 1) # coz decoder outputs an RGB image
                likes[module] = dist.log_prob(data[module]).astype(torch.float32)
                losses[module] = -likes[module].mean()
        model_loss = sum(losses.values())

        outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, kl=kl_value)

        metrics = {f'{name}_loss': value for name, value in losses.items()}
        metrics['model_kl'] = kl_value.mean()
        metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
        metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
        last_state = tuple([v[:, -1] for v in post.items()])

        return model_loss, last_state, outs, metrics
    
    def imagine(self, policy, start, is_terminal, horizon):
        #start is the posterior embedding, policy is an actor object
        start = [v.reshape((-1, v.shape[2:])) for v in start] #collapse the batch dimensions
        start['feat'] = self.rssm.get_feat(start)
        start['action'] = torch.zeros_like(policy(start['feat']).mode()) # take the most probable action  

        seq = {k: [v] for k, v in start.items()}
        for _ in range(horizon):
            action = policy(seq["feat"][-1].detach()).sample() # choose an action based on the state representation of the ast encountered state
            state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action) # imagine one-step on last state representation and last action chosen
            feat = self.rssm.get_feat(state) # get the features for this state

            for key, value in {**state, 'action': action, 'feat': feat}.items():
                seq[key].append(value) # append features to list
        seq = {k: torch.stack(v, 0) for k, v in seq.items()} #stack em

        disc = self.discount(seq['feat']).mean()
        if is_terminal is not None:
            # Override discount prediction for the first step with the true
            # discount factor from the replay buffer.
            true_first = 1.0 - is_terminal.reshape((-1, is_terminal.shape[2:])).astype(disc.dtype)
            disc = torch.concat([true_first[None], disc[1:]], 0)
        else:
            disc = torch.ones(seq['feat'].shape[:-1])
        seq['discount'] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        seq['weight'] = torch.cumprod(
            torch.concat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0) # discount propagation: yi​=x1​×x2​×x3​×⋯×xi​
        return seq
    
    def preprocess(self, obs):
        obs = obs.copy()
        # obs['reward'] = {
        # 'identity': lambda x: x,
        # 'sign': torch.sign,
        # 'tanh': torch.tanh,
        # }[self.config.clip_rewards](obs['reward'])
        obs["reward"] = torch.tanh(obs["reward"])
        obs['discount'] = 1.0 - obs['is_terminal']#.astype(dtype)
        return obs
    
    def video_pred(self, data, key):
        truth = data[key][:6] + 0.5
        embed = self.encoder(data)
        states, _ = self.rssm.observe(embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5])
        recon = self.decoder(self.rssm.get_feat(states))[key][:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data['action'][:6, 5:], init)
        openl = self.decoder(self.rssm.get_feat(prior))[key]
        model = torch.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = torch.concat([truth, model, error], 2)
        B, T, H, W, C = video.shape
        return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))