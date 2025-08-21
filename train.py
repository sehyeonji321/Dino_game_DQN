# train.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_params, save_params
from config import FINAL_EPSILON, INITIAL_EPSILON, EXPLORE, SAVE_INTERVAL, MODEL_DIR
from config import OBSERVATION, GAMMA, BATCH_SIZE, LEARNING_RATE, SCHEDULER_TYPE, SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA, SCHEDULER_MIN_LR, SCHEDULER_PATIENCE, SCHEDULER_FACTOR


def train_network(model, game_state, observe=False):
    params = load_params()
    D = params["D"]
    t = params["time"]
    epsilon = params["epsilon"]

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Add learning rate scheduler based on config
    if SCHEDULER_TYPE == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    elif SCHEDULER_TYPE == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHEDULER_GAMMA)
    elif SCHEDULER_TYPE == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, 
                                                        patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    
    loss_fn = nn.MSELoss()

    do_nothing = np.zeros(3)
    do_nothing[0] = 1

    x_t, r_0, terminal = game_state.get_state(do_nothing)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

    OBSERVE = 999999999 if observe else OBSERVATION
    
    # Initialize loss tracking variables
    loss_sum = 0
    episode_loss = 0
    training_steps = 0

    while True:
        a_t = np.zeros([3])

        if random.random() <= epsilon:
            action_index = np.random.choice([0,1,2], p=[0.45, 0.5, 0.05]) # ver.4: 엎드리는 경우는 의도적으로 낮게 sampling
            a_t[action_index] = 1
        else:
            q = model(torch.tensor(s_t).float())
            _, action_index = torch.max(q, 1)
            action_index = action_index.item()
            a_t[action_index] = 1

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        x_t1, r_t, terminal = game_state.get_state(a_t)
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

         # deque 자체가 maxlen 있어서 pop 불필요
        D.append((s_t, action_index, r_t, s_t1, terminal))

        if t > OBSERVE:
            minibatch = random.sample(D, BATCH_SIZE)
            inputs = np.zeros((BATCH_SIZE, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            targets = np.zeros((BATCH_SIZE, 3))

            for i in range(BATCH_SIZE):
                state_t, action_t, reward_t, state_t1, terminal = minibatch[i]
                inputs[i:i + 1] = state_t
                
                # Get current Q-values for the state
                with torch.no_grad():
                    current_q = model(torch.tensor(state_t).float()).numpy()[0]
                    next_q = model(torch.tensor(state_t1).float()).numpy()[0]
                
                # Create target Q-values
                target = current_q.copy()
                
                if terminal:
                    target[action_t] = reward_t
                else:
                    target[action_t] = reward_t + GAMMA * np.max(next_q)

                targets[i] = target

            outputs = model(torch.tensor(inputs).float())
            loss = loss_fn(outputs, torch.tensor(targets).float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Step the learning rate scheduler
            if SCHEDULER_TYPE == "ReduceLROnPlateau":
                # For ReduceLROnPlateau, we need to pass the loss value
                scheduler.step(loss.item())
            else:
                # For other schedulers, just step
                scheduler.step()

            # Update loss tracking
            loss_sum += loss.item()
            episode_loss += loss.item()
            training_steps += 1

        # 다음 상태 업데이트
        if terminal:
            # crash 했으면 초기 상태 다시 쌓기
            do_nothing = np.zeros(3)
            do_nothing[0] = 1
            x_t, _, _ = game_state.get_state(do_nothing)
            s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
            
            # Reset episode loss when episode ends
            episode_loss = 0
        else:
            s_t = s_t1

            
        t += 1

        if t % SAVE_INTERVAL == 0:
            game_state._game.pause()
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"episode_{t}.pth"))
            torch.save(model.state_dict(), "./latest.pth")
            save_params({"D": D, "time": t, "epsilon": epsilon})
            
            # Save current learning rate for monitoring
            current_lr = scheduler.get_last_lr()[0]
            print(f'Current learning rate: {current_lr:.2e}')
            
            game_state._game.resume()

        # Print learning rate every 1000 steps for monitoring
        if t % 1000 == 0:
            current_lr = scheduler.get_last_lr()[0]
            if training_steps > 0:
                avg_loss = loss_sum / training_steps
                print(f'timestep: {t}, epsilon: {round(epsilon, 3)}, action: {action_index}, reward: {r_t}, avg_loss: {round(avg_loss, 6)}, lr: {current_lr:.2e}')
            else:
                print(f'timestep: {t}, epsilon: {round(epsilon, 3)}, action: {action_index}, reward: {r_t}, loss: N/A (not training yet), lr: {current_lr:.2e}')
        else:
            if training_steps > 0:
                avg_loss = loss_sum / training_steps
                print(f'timestep: {t}, epsilon: {round(epsilon, 3)}, action: {action_index}, reward: {r_t}, avg_loss: {round(avg_loss, 6)}')
            else:
                print(f'timestep: {t}, epsilon: {round(epsilon, 3)}, action: {action_index}, reward: {r_t}, loss: N/A (not training yet)')
