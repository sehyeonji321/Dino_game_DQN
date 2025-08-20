# main.py

from collections import deque
from utils import save_params, load_model, load_params
from game import Game, DinoAgent, GameState
from model import DinoNet
from train import train_network
from config import INITIAL_EPSILON

def play_game(observe=False):
    params = load_params()
    # 만약 처음 실행이라면 초기화
    if params["time"] == 0 and len(params["D"]) == 0:
        params = {"D": deque(maxlen=50000), "time": 0, "epsilon": INITIAL_EPSILON}
        save_params(params)

    game = Game()
    agent = DinoAgent(game)
    game_state = GameState(agent, game)
    try:
        model = DinoNet()
        model = load_model(model)
        train_network(model, game_state, observe)
    except StopIteration:
        game.end()
        
if __name__ == "__main__":
    play_game(observe=False)
