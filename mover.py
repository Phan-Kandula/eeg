import pyautogui


class mover(object):
    """docstring for mover"""

    def __init__(self):
        super(mover, self).__init__()

    def move_right():
        pyautogui.moveRel(100, 0, 0.5)

    def move_left():
        pyautogui.moveRel(-100, 0, 0.5)

    def move_up():
        pyautogui.moveRel(0, -100, 0.5)

    def move_down():
        pyautogui.moveRel(0, 100, 0.5)

