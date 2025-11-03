import pygame
import threading

class AlertSystem:
    def __init__(self, sound_path="assets/sounds/alert.wav"):
        pygame.mixer.init()
        self.sound = pygame.mixer.Sound(sound_path)
        self.alert_active = False

    def trigger_alert(self):
        if not self.alert_active:
            self.alert_active = True
            # pornește sunetul într-un thread pentru a nu bloca
            threading.Thread(target=self.play_sound_loop, daemon=True).start()

    def play_sound_loop(self):
        self.sound.play(loops=-1)  # -1 = loop infinit

    def reset_alert(self):
        if self.alert_active:
            self.sound.stop()      # oprește imediat sunetul
            self.alert_active = False
