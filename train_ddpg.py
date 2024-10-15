import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(agent, env, options, train_options, beam_id):
    torch.autograd.set_detect_anomaly(True)
    CB_Env = env
    state = CB_Env.reset()

    iteration = 0
    num_of_iter = train_options['num_iter']
    total_reward = 0
    moving_avg_reward = 0
    while iteration < num_of_iter:
        try:
            action = agent.choose_action(state)
            next_state, reward, bf_gain, terminal = CB_Env.step(action)
            
            agent.update_replay_memory((state, action, reward, next_state, terminal))
            
            if len(agent.replay_memory) > agent.minibatch_size:
                qf1_loss, qf2_loss, policy_loss = agent.learn()
            else:
                qf1_loss, qf2_loss, policy_loss = 0, 0, 0
            
            state = next_state
            total_reward += reward.item() if isinstance(reward, torch.Tensor) else reward
            moving_avg_reward = 0.99 * moving_avg_reward + 0.01 * (reward.item() if isinstance(reward, torch.Tensor) else reward)
            iteration += 1
            train_options['overall_iter'] += 1

            if iteration % 10 == 0:
                avg_reward = total_reward / iteration
                bf_gain_value = bf_gain.item() if isinstance(bf_gain, torch.Tensor) else bf_gain
                logger.info(f"Beam: {beam_id}, Iter: {train_options['overall_iter']}, "
                            f"Avg Reward: {avg_reward:.4f}, "
                            f"Moving Avg Reward: {moving_avg_reward:.4f}, "
                            f"BF Gain: {bf_gain_value:.4f}, "
                            f"Q1 Loss: {qf1_loss:.4f}, Q2 Loss: {qf2_loss:.4f}, Policy Loss: {policy_loss:.4f}")

        except Exception as e:
            logger.error(f"Error occurred during training: {str(e)}")
            logger.error(traceback.format_exc())
            break

    train_options['state'] = state
    train_options['best_state'] = CB_Env.best_bf_vec

    return train_options
