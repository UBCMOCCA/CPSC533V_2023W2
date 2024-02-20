import gymnasium as gym
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_policy(policy, env='CartPole-v1', num_test_episodes=10, render=False, verbose=False):
    render_mode = "human" if render else None
    test_env = gym.make(env, render_mode=render_mode)
    test_rewards = []
    for i in range(num_test_episodes):
        state, _ = test_env.reset()
        episode_total_reward = 0
        while True:
            state = torch.tensor(np.array(state), device=device, dtype=torch.float32).unsqueeze(0)
            action = policy.select_action(state).cpu().numpy()[0][0]
            next_state, reward, terminated, truncated, _ = test_env.step(action)
            episode_total_reward += reward
            state = next_state
            if terminated | truncated:
                if verbose:
                    print('[Episode {:4d}/{}] [reward {:.1f}]'
                        .format(i, num_test_episodes, episode_total_reward))
                break
        test_rewards.append(episode_total_reward)
    test_env.close()
    return sum(test_rewards)/num_test_episodes


if __name__ == "__main__":
    import argparse
    from model import MyModel

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=None, type=str,
        help='Path to the model weights.')
    parser.add_argument('--env', default='CartPole-v1', type=str,
        help='Name of the environment.')
    
    args = parser.parse_args()
    env = gym.make(args.env)
    model = MyModel(state_size=len(env.reset()[0]), action_size=env.action_space.n)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    env.close()

    eval_policy(policy=model, env=args.env, render=False, verbose=True)
