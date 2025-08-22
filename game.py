# game.py

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from config import CHROME_DRIVER_PATH, GAME_URL, INIT_SCRIPT
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver import ActionChains ### ver.3 추가

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
        
        # self._driver.execute_script("Runner.config.ACCELERATION=0") # 게임 속도를 고정시키는 코드
        self._driver.execute_script(INIT_SCRIPT)

        # 키 입력을 위한 ActionChains 
        self.actions = ActionChains(self._driver) # ver.3 추가
    
    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
    
    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")
    
    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")
    
    def press_up(self):
        self._driver.find_element("tag name", "body").send_keys(Keys.ARROW_UP)
    
    # def press_down(self):
    #     self._driver.find_element("tag name", "body").send_keys(Keys.ARROW_DOWN)

     # ↓ 키를 '누르고 있는' 동작 ver.3
    def key_down(self, key):
        self.actions.key_down(key).perform()

    # ↓ 키를 '뗀다' ver.3
    def key_up(self, key):
        self.actions.key_up(key).perform()

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
        self.duck_frames = 0   # duck 상태 유지할 프레임 수
    
    def is_running(self):
        return self._game.get_playing()
    
    def is_crashed(self):
        return self._game.get_crashed()
    
    def jump(self):
        self._game.press_up()
    
    def duck(self, hold_frames):
        self._game.key_down(Keys.ARROW_DOWN)   # ↓ 누르기 시작
        self.duck_frames = hold_frames

    def update_duck(self):
        if self.duck_frames > 0:
            self.duck_frames -= 1
            if self.duck_frames == 0:
                self._game.key_up(Keys.ARROW_DOWN)  # ↓ 떼기

class GameState:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game
        self._display = show_img()
        self._display.__next__()
    
    def get_state(self, actions):
        score = self._game.get_score()
        reward = 0.2
        is_over = False
        
        # 만약 duck 유지 중이면 다른 액션은 무시
        if self._agent.duck_frames > 0:
            self._agent.update_duck()
            reward = -0.4
        else:
            if actions[1] == 1:      # jump
                self._agent.jump()
                reward = -0.4
            elif actions[2] == 1:    # duck 시작
                self._agent.duck(hold_frames=3)  # 0.25초 정도 유지
                reward = -0.4

        
        image = grab_screen(self._game._driver)
        self._display.send(image)
        
        if self._agent.is_crashed():
            self._game.restart()
            reward = -100
            is_over = True
        
        return image, reward, is_over
