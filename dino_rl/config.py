# config.py

import os
from collections import deque
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

GAME_URL = "chrome://dino"
CHROME_DRIVER_PATH = ChromeDriverManager().install()

DATA_DIR = "./data"
MODEL_DIR = "./model"
SAVE_INTERVAL = 1000

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

PARAMS_FILE = os.path.join(DATA_DIR, "params.pkl")

INIT_SCRIPT = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
GET_BASE64_SCRIPT = "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)"

ACTIONS = 3
GAMMA = 0.99
OBSERVATION = 1000  
EXPLORE = 500000  
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1  
REPLAY_MEMORY = 100000  
BATCH_SIZE = 32  
LEARNING_RATE = 1e-4  
IMG_CHANNELS = 4
IMG_SIZE = 128   # resolution 조절하기. 변경 시 model.py CNN FC layer 고려할 것!