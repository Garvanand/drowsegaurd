import collections
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
        self.current_level = 0
        self.level_history = collections.deque(maxlen=10)

    def _generate_beep(self, frequency, duration_s=0.4):
        t = np.linspace(0, duration_s, int(44100 * duration_s), endpoint=False)
        wave = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        stereo = np.column_stack((wave, wave))
        return stereo

    def _play_alert_loop(self):
        while self.alert_active:
            beep_freq = config.ALERT_CRITICAL_FREQ if self.current_level == 2 else config.ALERT_WARN_FREQ
            sleep_time = 0.5 if self.current_level == 2 else 1.0
            if self.mixer_ready:
                try:
                    if os.path.exists(config.ALERT_SOUND_FILE):
                        sound = pygame.mixer.Sound(config.ALERT_SOUND_FILE)
                    else:
                        raise FileNotFoundError(config.ALERT_SOUND_FILE)
                    sound.play()
                except Exception:
                    try:
                        sound = pygame.sndarray.make_sound(self._generate_beep(beep_freq))
                        sound.play()
                    except Exception:
                        pass
            time.sleep(sleep_time)
            time.sleep(0.1)

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

    def update(self, fatigue_score):
        new_level = 2 if fatigue_score >= config.FATIGUE_CRITICAL_THRESHOLD else 1 if fatigue_score >= config.FATIGUE_WARN_THRESHOLD else 0
        if new_level != self.current_level:
            self.current_level = new_level
            if new_level > 0 and not self.alert_active:
                self.trigger_alert()
            elif new_level == 0:
                self.stop_alert()
        self.level_history.append(new_level)

    def get_level_label(self):
        return ["normal", "warning", "critical"][self.current_level]

    def get_alert_duration(self):
        if self.drowsy_start_time is None:
            return 0.0
        return time.time() - self.drowsy_start_time
