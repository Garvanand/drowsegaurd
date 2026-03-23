import os
import threading
import time

import numpy as np
import pygame

import config


class AlertSystem:
    def __init__(self):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2)
            self.mixer_ready = True
        except Exception:
            self.mixer_ready = False
        self.alert_active = False
        self.alert_thread = None
        self.drowsy_start_time = None

    def _generate_beep_sound(self):
        if not self.mixer_ready:
            return None
        t = np.linspace(0, 0.5, int(44100 * 0.5))
        wave = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        stereo = np.column_stack((wave, wave))
        return pygame.sndarray.make_sound(stereo)

    def _play_alert_loop(self):
        fallback = None
        while self.alert_active:
            if not self.mixer_ready:
                time.sleep(0.5)
                continue
            try:
                if os.path.exists(config.ALERT_SOUND_FILE):
                    sound = pygame.mixer.Sound(config.ALERT_SOUND_FILE)
                else:
                    if fallback is None:
                        fallback = self._generate_beep_sound()
                    sound = fallback
                if sound is not None:
                    sound.play()
                    time.sleep(0.55)
                else:
                    time.sleep(0.5)
            except Exception:
                time.sleep(0.5)

    def trigger_alert(self):
        if self.alert_active:
            return
        self.alert_active = True
        if self.drowsy_start_time is None:
            self.drowsy_start_time = time.time()
        self.alert_thread = threading.Thread(target=self._play_alert_loop, daemon=True)
        self.alert_thread.start()

    def stop_alert(self):
        self.alert_active = False
        if self.mixer_ready:
            try:
                pygame.mixer.stop()
            except Exception:
                pass
        self.drowsy_start_time = None

    def update(self, is_drowsy):
        if is_drowsy and not self.alert_active:
            self.trigger_alert()
        if not is_drowsy and self.alert_active:
            self.stop_alert()

    def get_alert_duration(self):
        if self.drowsy_start_time is None:
            return 0
        return time.time() - self.drowsy_start_time
