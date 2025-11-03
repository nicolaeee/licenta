import winsound

class Alerts:
    def __init__(self):
        pass

    def beep(self):
        frequency = 1500
        duration = 500
        winsound.Beep(frequency, duration)
