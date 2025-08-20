# game.py

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from config import CHROME_DRIVER_PATH, GAME_URL, INIT_SCRIPT
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from utils import grab_screen, show_img

class Game:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        service = Service(CHROME_DRIVER_PATH)
        self._driver = webdriver.Chrome(service=service, options=chrome_options)
        self._driver.set_window_position(x=300, y=300)
        self._driver.set_window_size(900, 600)
        
        try : 
            self._driver.get(GAME_URL)
        except:
            pass
        
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(INIT_SCRIPT)
    
    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
    
    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")
    
    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")
    
    def press_up(self):
        self._driver.find_element("tag name", "body").send_keys(Keys.ARROW_UP)
    
    def press_down(self):
        self._driver.find_element("tag name", "body").send_keys(Keys.ARROW_DOWN)
    
    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        return int(''.join(score_array))
    
    def pause(self):
        self._driver.execute_script("return Runner.instance_.stop()")
    
    def resume(self):
        self._driver.execute_script("return Runner.instance_.play()")
    
    def end(self):
        self._driver.close()

class DinoAgent:
    def __init__(self, game):
        self._game = game
        self.jump()
    
    def is_running(self):
        return self._game.get_playing()
    
    def is_crashed(self):
        return self._game.get_crashed()
    
    def jump(self):
        self._game.press_up()
    
    def duck(self):
        self._game.press_down()

class GameState:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game
        self._display = show_img()
        self._display.__next__()
    
    def get_state(self, actions):
        score = self._game.get_score()
        reward = 0.1
        is_over = False
        
        if actions[1] == 1:
            self._agent.jump()
            reward = -0.01
        
        image = grab_screen(self._game._driver)
        self._display.send(image)
        
        if self._agent.is_crashed():
            self._game.restart()
            reward = -10
            is_over = True
        
        return image, reward, is_over
