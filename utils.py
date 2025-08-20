# utils.py

import os
import pickle
import base64
from io import BytesIO
from collections import deque

import numpy as np
import cv2
from PIL import Image

from config import PARAMS_FILE, GET_BASE64_SCRIPT, IMG_SIZE
import torch

def save_params(params):
    with open(PARAMS_FILE, 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

def load_params():
    if os.path.isfile(PARAMS_FILE):
        with open(PARAMS_FILE, 'rb') as f:
            return pickle.load(f)
    return {
        "D": deque(maxlen=50000),
        "time": 0,
        "epsilon": 0.01
    }

def load_model(model):
    if os.path.isfile('./latest.pth'):
        model.load_state_dict(torch.load('./latest.pth'))
    return model

def grab_screen(driver):
    image_b64 = driver.execute_script(GET_BASE64_SCRIPT)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    return process_img(screen)

def process_img(image): ####################################################
    # RGBA 방어코드
    if image.ndim == 3 and image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 3:  # RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 이미지 리사이즈
    image = image[:300, :500]
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return image


def show_img(graphs=False):
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        imS = cv2.resize(screen, (800, 400))
        # cv2.imshow(window_title, screen)
        cv2.imshow(window_title, imS)   # imS로 수정
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
