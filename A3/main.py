import time

import numpy as np
import gymnasium as gym
from gymnasium.spaces.discrete import Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from utils import count_vars, discount_cumsum, args_to_str
from models import ActorCritic
from pg_buffer import PGBuffer

from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
#import PIL

def main(args):
    # create environment 
    env = gym.make(args.env, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, Discrete):
        discrete = True
        act_dim = env.action_space.n
    else:
        discrete = False
        act_dim = env.action_space.shape[0]

    # actor critic 
    ac = ActorCritic(obs_dim, act_dim, discrete).to(args.device)
    print('Number of parameters', count_vars(ac))

    # Set up experience buffer
    steps_per_epoch = int(args.steps_per_epoch)
    buf = PGBuffer(obs_dim, act_dim, discrete, steps_per_epoch, args)
    logs = defaultdict(lambda: [])
    writer = SummaryWriter(args_to_str(args))
    gif_frames = []

    # Set up function for computing policy loss
    def compute_loss_pi(batch):
        obs, act, psi, logp_old = batch['obs'], batch['act'], batch['psi'], batch['logp']
        pi, logp = ac.pi(obs, act)

        # Policy loss
        if args.loss_mode == 'vpg':
            # TODO (Task 2): implement vanilla policy gradient loss
        elif args.loss_mode == 'ppo':
            # TODO (Task 4): implement clipped PPO loss
        else:
            raise Exception('Invalid loss_mode option', args.loss_mode)

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(batch):
        obs, ret = batch['obs'], batch['ret']
        v = ac.v(obs)
        # TODO: (Task 2): compute value function loss
        return loss_v

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=args.pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=args.v_lr)

    # Set up update function
    def update():
        batch = buf.get()

        # Get loss and info values before update
        pi_l_old, pi_info_old = compute_loss_pi(batch)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(batch).item()

        # Policy learning
        for i in range(args.train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(batch)
            loss_pi.backward()
            pi_optimizer.step()

        # Value function learning
        for i in range(args.train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(batch)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logs['kl'] += [kl]
        logs['ent'] += [ent]
        logs['loss_v'] += [loss_v.item()]
        logs['loss_pi'] += [loss_pi.item()]

    # Prepare for interaction with environment
    start_time = time.time()
    o, _ = env.reset(seed=args.seed)
    ep_ret, ep_len = 0, 0

    ep_count = 0  # just for logging purpose, number of episodes run
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(args.epochs):
        for t in range(steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).to(args.device))

            next_o, r, d, truncated, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            if ep_count % 100 == 0:
                frame = env.render()
                # uncomment this line if you want to log to tensorboard (can be memory intensive)
                #gif_frames.append(frame)
                #gif_frames.append(PIL.Image.fromarray(frame).resize([64,64]))  # you can try this downsize version if you are resource constrained
                time.sleep(0.01)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == args.max_ep_len
            terminal = d or truncated or timeout
            epoch_ended = t==steps_per_epoch-1

            if terminal or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32).to(args.device))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logs['ep_ret'] += [ep_ret]
                    logs['ep_len'] += [ep_len]
                    ep_count += 1

                o, _ = env.reset()
                ep_ret, ep_len = 0, 0

                # save a video to tensorboard so you can view later
                if len(gif_frames) != 0:
                    vid = np.stack(gif_frames)
                    vid_tensor = vid.transpose(0,3,1,2)[None]
                    writer.add_video('rollout', vid_tensor, epoch, fps=50)
                    gif_frames = []
                    writer.flush()
                    print('wrote video')

        # Perform VPG update!
        update()

        if epoch % 10 == 0:
            vals = {key: np.mean(val) for key, val in logs.items()}
            for key in vals:
                writer.add_scalar(key, vals[key], epoch)
            writer.flush()
            print('Epoch', epoch, vals)
            logs = defaultdict(lambda: [])



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='LunarLander-v2', help='[CartPole-v0, LunarLander-v2, LunarLanderContinuous-v2, others]')

    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to run')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lam', type=float, default=0.97, help='GAE-lambda factor')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help='Number of env steps to run during optimizations')
    parser.add_argument('--max_ep_len', type=int, default=1000)

    parser.add_argument('--train_pi_iters', type=int, default=4)
    parser.add_argument('--train_v_iters', type=int, default=40)
    parser.add_argument('--pi_lr', type=float, default=1e-3, help='Policy learning rate')
    parser.add_argument('--v_lr', type=float, default=3e-4, help='Value learning rate')

    parser.add_argument('--psi_mode', type=str, default='gae', help='value to modulate logp gradient with [future_return, gae]')
    parser.add_argument('--loss_mode', type=str, default='vpg', help='Loss mode [vpg, ppo]')
    parser.add_argument('--clip_ratio', type=float, default=0.1, help='PPO clipping ratio')

    parser.add_argument('--render_interval', type=int, default=100, help='render every N')
    parser.add_argument('--log_interval', type=int, default=100, help='log every N')

    parser.add_argument('--device', type=str, default='cpu', help='you can set this to cuda if you have a GPU')

    parser.add_argument('--suffix', type=str, default='', help='Just for experiment logging (see utils)')
    parser.add_argument('--prefix', type=str, default='logs', help='Just for experiment logging (see utils)')
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
