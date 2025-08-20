# main.py

from collections import deque
from utils import save_params, load_model
from game import Game, DinoAgent, GameState
from model import DinoNet
from train import train_network

def play_game(observe=False):
    params = {"D": deque(maxlen=50000), "time": 0, "epsilon": 0.001}
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
