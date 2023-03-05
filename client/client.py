import readchar
import json
import time
import os
import threading
import websocket

from common import preinit_tensorflow, display_image, import_labels, decode_decompressed_image, input_from_image
if __name__ == "__main__":
    preinit_tensorflow()
import tensorflow as tf
from runner import ModelRunner

MODEL = "botwqt.tflite"
CONFIDENCE_THRESHOLD = 100
PORT = "8899"

COLOR_RESET = "\033[0m"
COLOR_SELECTED = "\033[1;33m"
HINT = "Use arrow keys to navigate and Enter to mark/unmark"
UP = "\u25B2"
DOWN = "\u25BC"
class Screen:
    lock: threading.Lock
    image: str
    current_column: int
    discovered_list: list[str]
    undiscovered_list: list[str]
    discovered_current_row: int
    undiscovered_current_row: int
    discovered_display_start: int
    undiscovered_display_start: int
    cols: int
    rows: int
    quest_rows: int
    def __init__(self):
        self.lock = threading.RLock()
        self.cols = 0
        self.rows = 0
        self.image = "Starting..."
        self.current_column = 0
        self.discovered_list = []
        self.undiscovered_list = []
        self.discovered_current_row = 0
        self.undiscovered_current_row = 0
        self.discovered_display_start = 0
        self.undiscovered_display_start = 0
        self.quest_rows = 0

    def update(self, redraw_image=False, redraw_discovered_quests=False, redraw_undiscovered_quests=False):
        if self.current_column == 0 and not self.discovered_list:
            self.current_column = 1
        elif self.current_column == 1 and not self.undiscovered_list:
            self.current_column = 0
        if self.current_column == 0:
            if self.discovered_current_row >= len(self.discovered_list):
                self.discovered_current_row = len(self.discovered_list) - 1
            if self.discovered_current_row < 0:
                self.discovered_current_row = 0
        else:
            if self.undiscovered_current_row >= len(self.undiscovered_list):
                self.undiscovered_current_row = len(self.undiscovered_list) - 1
            if self.undiscovered_current_row < 0:
                self.undiscovered_current_row = 0
        self.lock.acquire()
        try:
            terminal_size = os.get_terminal_size()
            if self.cols != terminal_size.columns or self.rows != terminal_size.lines:
                print("\033[2J")
                self.cols = terminal_size.columns
                self.rows = terminal_size.lines
                redraw_image = True
                redraw_discovered_quests = True
                redraw_undiscovered_quests = True
            image_rows = len(self.image.split("\n"))
            quest_row_start = image_rows + 2
            quest_rows = self.rows - 5 - image_rows
            if redraw_image and quest_rows != self.quest_rows:
                self.quest_rows = quest_rows
                redraw_discovered_quests = True
                redraw_undiscovered_quests = True
            half_width = self.cols // 2


            if redraw_image:
                print(f"\033[1;0H{self.image}")
            
            
            if redraw_discovered_quests:
                print(f"\033[{image_rows+1};0H{self.fit(f'Discovered ({len(self.discovered_list)}):', half_width)}")
                
                if self.discovered_display_start > 0:
                    print(f"\033[{image_rows+2};0H{self.fit(UP, half_width, center=True)}")
                else:
                    print(f"\033[{image_rows+2};0H{self.fit('', half_width)}")
                
                for r in range(self.quest_rows):
                    quest_index = self.discovered_display_start + r
                    row_index = quest_row_start + r
                    if quest_index < len(self.discovered_list):
                        quest_name = self.discovered_list[quest_index]
                        if self.current_column == 0 and quest_index == self.discovered_current_row:
                            print(f"\033[{row_index+1};0H{COLOR_SELECTED}{self.fit(f' > {quest_name}', half_width)}{COLOR_RESET}")
                        else:
                            print(f"\033[{row_index+1};0H{self.fit(f'   {quest_name}', half_width)}")
                    else:
                        print(f"\033[{row_index+1};0H{self.fit('', half_width)}")
                if self.discovered_display_start + self.quest_rows < len(self.discovered_list):
                    print(f"\033[{self.rows-2};0H{self.fit(DOWN, half_width, center=True)}")
                else:
                    print(f"\033[{self.rows-2};0H{self.fit('', half_width)}")

            if redraw_undiscovered_quests:
                print(f"\033[{image_rows+1};{half_width+1}H{self.fit(f'Undiscovered ({len(self.undiscovered_list)}):', half_width)}")
                if self.undiscovered_display_start > 0:
                    print(f"\033[{image_rows+2};{half_width+1}H{self.fit(UP, half_width, center=True)}")
                else:
                    print(f"\033[{image_rows+2};{half_width+1}H{self.fit('', half_width)}")
                for r in range(self.quest_rows):
                    quest_index = self.undiscovered_display_start + r
                    row_index = quest_row_start + r
                    if quest_index < len(self.undiscovered_list):
                        quest_name = self.undiscovered_list[quest_index]
                        if self.current_column != 0 and quest_index == self.undiscovered_current_row:
                            print(f"\033[{row_index+1};{half_width+1}H{COLOR_SELECTED}{self.fit(f' > {quest_name}', half_width)}{COLOR_RESET}")
                        else:
                            print(f"\033[{row_index+1};{half_width+1}H{self.fit(f'   {quest_name}', half_width)}")
                    else:
                        print(f"\033[{row_index+1};{half_width+1}H{self.fit('', half_width)}")
                if self.undiscovered_display_start + self.quest_rows < len(self.undiscovered_list):
                    print(f"\033[{self.rows-2};{half_width+1}H{self.fit(DOWN, half_width, center=True)}")
                else:
                    print(f"\033[{self.rows-2};{half_width+1}H{self.fit('', half_width)}")
            print(f"\033[{self.rows-1};0H{self.fit(HINT, self.cols)}")
        finally:
            self.lock.release()

    def key_up(self):
        if self.current_column == 0:
            if self.discovered_current_row > 0:
                self.discovered_current_row -= 1
                if self.discovered_current_row < self.discovered_display_start:
                    self.discovered_display_start -= 1
            else:
                self.discovered_current_row = 0
            self.update(redraw_discovered_quests=True)
        else:
            if self.undiscovered_current_row > 0:
                self.undiscovered_current_row -= 1
                if self.undiscovered_current_row < self.undiscovered_display_start:
                    self.undiscovered_display_start -= 1
            else:
                self.undiscovered_current_row = 0
            self.update(redraw_undiscovered_quests=True)

    def key_down(self):
        if self.current_column == 0:
            if self.discovered_current_row < len(self.discovered_list)-1:
                self.discovered_current_row += 1
                if self.discovered_current_row >= self.discovered_display_start + self.quest_rows:
                    self.discovered_display_start += 1
            else:
                self.discovered_current_row = len(self.discovered_list)-1
            self.update(redraw_discovered_quests=True)
        else:
            if self.undiscovered_current_row < len(self.undiscovered_list)-1:
                self.undiscovered_current_row += 1
                if self.undiscovered_current_row >= self.undiscovered_display_start + self.quest_rows:
                    self.undiscovered_display_start += 1
            else:
                self.undiscovered_current_row = len(self.undiscovered_list)-1
            self.update(redraw_undiscovered_quests=True)

    def key_left(self):
        if self.current_column != 0:
            self.current_column = 0
            self.update(redraw_undiscovered_quests=True, redraw_discovered_quests=True)
    
    def key_right(self):
        if self.current_column != 1:
            self.current_column = 1
            self.update(redraw_undiscovered_quests=True, redraw_discovered_quests=True)

    def fit(self, text, size, center=False):
        if center and len(text) < size:
            text = " "*(size//2-len(text)//2) + text
        if len(text) > size:
            return text[:size-3] + "..."
        if len(text) < size:
            return text + " "*(size-len(text))
        return text

class Runtime:
    quests: list[str]
    discovered: list[bool]

    def __init__(self):
        self.quests = import_labels(captialize=True)
        self.discovered = [False] * len(self.quests)
        if os.path.isfile("state.json"):
            with open("state.json", "r") as f:
                self.discovered = json.load(f)
    
    def mark(self, idx):
        self.discovered[idx] = True
        with open("state.json", "w") as f:
            json.dump(self.discovered, f)

    def toggle(self, quest):
        idx = self.quests.index(quest)
        self.discovered[idx] = not self.discovered[idx]
        with open("state.json", "w") as f:
            json.dump(self.discovered, f)
    
    def get_lists(self):
        discovered_list = []
        undiscovered_list = []
        for i in range(1, len(self.quests)):
            if self.discovered[i]:
                discovered_list.append(self.quests[i])
            else:
                undiscovered_list.append(self.quests[i])
        discovered_list.sort()
        undiscovered_list.sort()
        return discovered_list, undiscovered_list

class Client:
    screen: Screen
    runtime: Runtime
    port: int
    
    def __init__(self, screen, runtime, port):
        self.screen = screen
        self.runtime = runtime
        self.port = port

    def run(self):
        runner = ModelRunner(MODEL)
        self.screen.image = self.screen.fit("Waiting for connection...", self.screen.cols)
        self.screen.update(redraw_image=True)
        ws = websocket.WebSocket()
        while True:
            
            while True:
                try:
                    ws.connect(f"ws://localhost:{self.port}/")
                    break
                except:
                    time.sleep(1)
            try:
                while True:
                    message = ws.recv()
                    try:
                        image = decode_decompressed_image(bytes.fromhex(message))
                        quest, confidence = runner.run_one(input_from_image(image))
                        image = display_image(image)
                        if confidence < CONFIDENCE_THRESHOLD:
                            quest = 0
                    except:
                        image = self.screen.fit(message, self.screen.cols)
                        quest = 0

                    self.screen.image = image
                    if quest != 0:
                        runtime.mark(quest)
                        self.screen.discovered_list, self.screen.undiscovered_list = runtime.get_lists()
                        self.screen.update(redraw_image=True, redraw_discovered_quests=True, redraw_undiscovered_quests=True)
                    else:
                        self.screen.update(redraw_image=True)
            except:
                self.screen.image = self.screen.image = self.screen.fit("Reconnecting...", self.screen.cols)
                self.screen.update(redraw_image=True)


    
if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    runtime = Runtime()
    screen = Screen()
    client = Client(screen, runtime, PORT)
    client_thread = threading.Thread(target=client.run)
    client_thread.daemon = True
    client_thread.start()
    try:
        screen.discovered_list, screen.undiscovered_list = runtime.get_lists()
        screen.update(redraw_image=True, redraw_discovered_quests=True, redraw_undiscovered_quests=True)
        while True:
            k = readchar.readkey()
            if k == readchar.key.UP:
                screen.key_up()
            elif k == readchar.key.DOWN:
                screen.key_down()
            elif k == readchar.key.LEFT:
                screen.key_left()
            elif k == readchar.key.RIGHT:
                screen.key_right()
            elif k == readchar.key.ENTER:
                if screen.current_column == 0:
                    if screen.discovered_list:
                        runtime.toggle(screen.discovered_list[screen.discovered_current_row])
                        screen.discovered_list, screen.undiscovered_list = runtime.get_lists()
                        screen.update(redraw_discovered_quests=True, redraw_undiscovered_quests=True)
                else:
                    if screen.undiscovered_list:
                        runtime.toggle(screen.undiscovered_list[screen.undiscovered_current_row])
                        screen.discovered_list, screen.undiscovered_list = runtime.get_lists()
                        screen.update(redraw_discovered_quests=True, redraw_undiscovered_quests=True)
    except KeyboardInterrupt:
        pass
