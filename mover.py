import pyautogui


class Mover(object):
    """docstring for Mover"""

    def __init__(self):
        super(Mover, self).__init__()

    def move_right(self):
        pyautogui.moveRel(100, 0, 0.5)

    def move_left(self):
        pyautogui.moveRel(-100, 0, 0.5)

    def move_up(self):
        pyautogui.moveRel(0, -100, 0.5)

    def move_down(self):
        pyautogui.moveRel(0, 100, 0.5)
