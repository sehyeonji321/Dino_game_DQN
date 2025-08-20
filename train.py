# train.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_params, save_params
from config import FINAL_EPSILON, INITIAL_EPSILON, EXPLORE, SAVE_INTERVAL, MODEL_DIR
from config import OBSERVATION, GAMMA, BATCH_SIZE, LEARNING_RATE


def train_network(model, game_state, observe=False):
    params = load_params()
    D = params["D"]
    t = params["time"]
    epsilon = params["epsilon"]

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    do_nothing = np.zeros(3)
    do_nothing[0] = 1

    x_t, r_0, terminal = game_state.get_state(do_nothing)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

    OBSERVE = 999999999 if observe else OBSERVATION

    while True:
        loss_sum = 0
        a_t = np.zeros([3])

        if random.random() <= epsilon:
            action_index = random.randrange(3)
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
                target = model(torch.tensor(state_t).float()).detach().numpy()[0]
                Q_sa = model(torch.tensor(state_t1).float()).detach().numpy()[0]

                if terminal:
                    target[action_t] = reward_t
                else:
                    target[action_t] = reward_t + GAMMA * np.max(Q_sa)

                targets[i] = target

            outputs = model(torch.tensor(inputs).float())
            loss = loss_fn(outputs, torch.tensor(targets).float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        # s_t = s_t1 if not terminal else s_t

        # 다음 상태 업데이트
        if terminal:
            # crash 했으면 초기 상태 다시 쌓기
            do_nothing = np.zeros(3)
            do_nothing[0] = 1
            x_t, _, _ = game_state.get_state(do_nothing)
            s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
        else:
            s_t = s_t1

            
        t += 1

        if t % SAVE_INTERVAL == 0:
            game_state._game.pause()
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"episode_{t}.pth"))
            torch.save(model.state_dict(), "./latest.pth")
            save_params({"D": D, "time": t, "epsilon": epsilon})
            game_state._game.resume()

        print(f'timestep: {t}, epsilon: {round(epsilon, 3)}, action: {action_index}, reward: {r_t}, loss: {round(loss_sum, 3)}')
