import speech_recognition as sr
import pyaudio
import logging
import time

# --- Logging Setup ---
logger = logging.getLogger("VCListener")
logger.setLevel(logging.INFO)
f_handler = logging.FileHandler('vc_listener.log', mode='a', encoding='utf-8')
f_handler.setLevel(logging.DEBUG)
f_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.handlers.clear()
logger.addHandler(f_handler)

def run_vc_listener(user_id, audio_queue):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        logger.info(f"Listener started for user {user_id}")

        while True:
            try:
                # Wait for speech to start, then record until silence
                logger.debug(f"User {user_id}: Waiting for speech...")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)
                # timeout=None: Waits forever for speech to start
                # phrase_time_limit=None: Records until silence is detected

                # Use the large Whisper model for better accuracy
                text = recognizer.recognize_whisper(audio, model="large")
                if text.strip():
                    logger.info(f"User {user_id} said: '{text}'")
                    audio_queue.put((user_id, text))
                else:
                    logger.debug(f"User {user_id}: Empty or silent audio detected")

            except sr.UnknownValueError:
                logger.debug(f"User {user_id}: Couldn’t understand audio")
                continue
            except Exception as e:
                logger.error(f"Error in listener for user {user_id}: {e}")
                continue

if __name__ == "__main__":
    # Quick test if you wanna run it standalone
    from queue import Queue
    audio_queue = Queue()
    run_vc_listener("test_user", audio_queue)