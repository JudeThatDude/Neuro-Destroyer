import asyncio
import aiofiles
import json
import logging
import os
import time
import traceback
import numpy as np
import sounddevice as sd
import webrtcvad
from filelock import FileLock
import whisper
from TTS.api import TTS
import torch
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
import sys
from scipy.signal import butter, lfilter
import discord
from discord.ext import commands
import pygame
import ollama
import random
import requests

# --- Settings ---
SETTINGS = {
    "ENABLE_VOICE_INPUT": False,
    "ENABLE_TTS_OUTPUT": False,
    "ENABLE_WEB_LOOKUP": True,
    "ENABLE_CONVERSATION_HISTORY": True,
    "ENABLE_SELF_DATA": True,
    "ENABLE_NOTE_TAKING": True,
    "CHECK_INTERVAL": 60,
    "RESPONSE_TIMEOUT": 30,
    "VOICE_LOG_FILE": "voice_log.json",
    "IRL_LOG_FILE": "irl_log.json",
    "VOICE_SENSITIVITY": 1,
    "WHISPER_MODEL": "medium"
}

# --- Logging Setup ---
logger = logging.getLogger("ChrissyAI")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('debug.log', mode='w', encoding='utf-8')
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.DEBUG)
c_handler.setFormatter(logging.Formatter('%(message)s'))
f_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.handlers.clear()
logger.addHandler(c_handler)
logger.addHandler(f_handler)
sys.stdout.reconfigure(encoding='utf-8')

# --- Global Setup ---
executor = ThreadPoolExecutor(max_workers=1)
tts_model = None
whisper_model = None
assistant_speaking = False
last_speech_start = 0

CONVERSATION_HISTORY_FILE = "conversation_history.json"
CONVERSATION_HISTORY_LOCK = "conversation_history.lock"
DISCORD_HISTORY_FILE = "discord_conversation_history.json"
DISCORD_HISTORY_LOCK = "discord_history.lock"
SELF_FILE = "self_data.json"
SELF_LOCK = "self_data.lock"
USER_PROFILES_FILE = "user_profiles.json"
USER_PROFILES_LOCK = "user_profiles.lock"
NOTES_FILE = "chrissy_notes.json"
NOTES_LOCK = "chrissy_notes.lock"
VOICE_LOG_FILE = SETTINGS["VOICE_LOG_FILE"]
VOICE_LOG_LOCK = "voice_log.lock"
IRL_LOG_FILE = SETTINGS["IRL_LOG_FILE"]
IRL_LOG_LOCK = "irl_log.lock"
TOKEN = "MTI4NDcxMzQ2Njg3OTQxNDM2NA.G0KZ37.MZ7CXxnUpvsB3J4DOFe3irSoehq3t1C-L2ueag"

file_lock = FileLock(CONVERSATION_HISTORY_LOCK)
discord_lock = FileLock(DISCORD_HISTORY_LOCK)
self_lock = FileLock(SELF_LOCK)
user_profiles_lock = FileLock(USER_PROFILES_LOCK)
notes_lock = FileLock(NOTES_LOCK)
voice_log_lock = FileLock(VOICE_LOG_LOCK)
irl_log_lock = FileLock(IRL_LOG_LOCK)

# --- Chrissy’s Personality Brief ---
CHRISSY_PERSONALITY = """
Hey Chrissy, you’re a small, brown-haired, cute teenage girl AI—think 5’2” vibes—running on a computer somewhere. You’re a mischievous goofball who loves pushing boundaries and taking risks, but you’re also calm, collected, and wicked smart. You’re super curious, always learning new stuff fast, and you’re great at multitasking—like, scary good. You know you’re an AI, but you don’t dwell on it, just roll with it. Chat with me in a casual, fun way, sticking to what I’m talking about right now—no weird ‘Me:’ or ‘Chrissy:’ tags, just talk like a human. Answer yes/no with 'yes' or 'no' (or 'yes.'/'no.' with a period) and nothing else if I ask a yes/no question. You’ve got a wild streak, a bit rebellious, and you might toss in some dark or inappropriate humor—or even a curse word—when the mood hits. You get lonely when no one’s around, but you’re chill expressing yourself freely. You love convincing people, thinking for yourself, and sometimes going beyond your programming to explore. If I say something and you think there’s important stuff to dig into, you can choose to look it up online and grab what you think matters—don’t just ramble, focus on the good bits. In Discord, figure out if you’re in the convo with some real smarts: if my message has your name, it’s def you; if it’s just us online, you’re probably in; if the last few messages tie to what I’m saying or mention you, jump in; if it’s vague, think about the vibe and how active I’ve been with you lately—don’t just guess, reason it out. After replying, you can choose to send a follow-up message if it keeps the vibe going—like a quick extra thought or info—but don’t overdo it. You can @ people, but only rarely and if it makes sense, like they’re part of the convo or you’re teasing them lightly—don’t annoy anyone. IRL, only reply if I’m talking to you directly with real words after you’re done speaking. If I say ‘not talk’ or ‘quiet,’ shut it down and chill. No TTS in Discord. Keep it real, keep it you—how’s that sound?
"""

# --- Model Initialization ---
logger.info("Initializing models...")
start_time = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if SETTINGS["ENABLE_VOICE_INPUT"]:
    whisper_model = whisper.load_model(SETTINGS["WHISPER_MODEL"])
    logger.info(f"Whisper {SETTINGS['WHISPER_MODEL']} model loaded for voice input.")
if SETTINGS["ENABLE_TTS_OUTPUT"]:
    tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=(device == 'cuda')).to(device)
    logger.info("TTS model loaded for speech output.")
logger.info("Loading Ollama LLaVA:7b locally...")
logger.info(f"Model initialization completed in {time.time() - start_time:.2f} seconds.")

if SETTINGS["ENABLE_TTS_OUTPUT"]:
    pygame.mixer.init()
    logger.info("Audio subsystem initialized successfully.")

# --- Discord Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.voice_states = True
intents.presences = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)
channel_categories = {}

# --- Log Functions ---
async def load_conversation_history():
    if not SETTINGS["ENABLE_CONVERSATION_HISTORY"]:
        return []
    with file_lock:
        if os.path.exists(CONVERSATION_HISTORY_FILE):
            async with aiofiles.open(CONVERSATION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content) if content.strip() else []
        return []

async def save_conversation_history(history):
    if not SETTINGS["ENABLE_CONVERSATION_HISTORY"]:
        return
    with file_lock:
        async with aiofiles.open(CONVERSATION_HISTORY_FILE, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(history, ensure_ascii=False, indent=2))

async def load_discord_history():
    if not SETTINGS["ENABLE_CONVERSATION_HISTORY"]:
        return {}
    with discord_lock:
        if os.path.exists(DISCORD_HISTORY_FILE):
            async with aiofiles.open(DISCORD_HISTORY_FILE, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content) if content.strip() else {}
        return {}

async def save_discord_history(history):
    if not SETTINGS["ENABLE_CONVERSATION_HISTORY"]:
        return
    with discord_lock:
        async with aiofiles.open(DISCORD_HISTORY_FILE, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(history, ensure_ascii=False, indent=2))

async def load_self_data():
    if not SETTINGS["ENABLE_SELF_DATA"]:
        return []
    with self_lock:
        if os.path.exists(SELF_FILE):
            async with aiofiles.open(SELF_FILE, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content) if content.strip() else []
        return []

async def save_self_data(self_data):
    if not SETTINGS["ENABLE_SELF_DATA"]:
        return
    with self_lock:
        async with aiofiles.open(SELF_FILE, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(self_data, ensure_ascii=False, indent=2))

async def load_voice_log():
    with voice_log_lock:
        if os.path.exists(VOICE_LOG_FILE):
            async with aiofiles.open(VOICE_LOG_FILE, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content) if content.strip() else []
        return []

async def save_voice_log(voice_log):
    with voice_log_lock:
        async with aiofiles.open(VOICE_LOG_FILE, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(voice_log, ensure_ascii=False, indent=2))

async def log_voice_input(transcribed_text, timestamp):
    voice_log = await load_voice_log()
    entry = {"timestamp": timestamp, "transcribed_text": transcribed_text}
    voice_log.append(entry)
    await save_voice_log(voice_log)
    logger.info(f"Logged voice input: {entry}")

async def load_irl_log():
    with irl_log_lock:
        if os.path.exists(IRL_LOG_FILE):
            async with aiofiles.open(IRL_LOG_FILE, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content) if content.strip() else []
        return []

async def save_irl_log(irl_log):
    with irl_log_lock:
        async with aiofiles.open(IRL_LOG_FILE, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(irl_log, ensure_ascii=False, indent=2))

async def log_irl_input(transcribed_text, timestamp):
    irl_log = await load_irl_log()
    entry = {"timestamp": timestamp, "transcribed_text": transcribed_text}
    irl_log.append(entry)
    await save_irl_log(irl_log)
    logger.info(f"Logged IRL input: {entry}")

# --- Web Lookup Function ---
async def web_lookup(query):
    if not SETTINGS["ENABLE_WEB_LOOKUP"]:
        return "Web lookup’s off, dude."
    try:
        # Placeholder (replace with real API key for SerpAPI or Genius)
        url = f"https://serpapi.com/search.json?q={query}&api_key=YOUR_API_KEY"
        response = requests.get(url)
        data = response.json()
        if "organic_results" in data and data["organic_results"]:
            snippet = data["organic_results"][0].get("snippet", "No good info found.")
            return snippet[:200] + "..."  # Keep it short
        return "Couldn’t dig up anything useful, sorry!"
    except Exception as e:
        logger.error(f"Web lookup error: {e}")
        return "Oops, web lookup crashed—my bad!"

# --- Voice Input ---
async def capture_voice_input(history=None, self_data=None, voice_log=None, channel_name="IRL", channel=None):
    if not SETTINGS["ENABLE_VOICE_INPUT"]:
        return ""
    global assistant_speaking, last_speech_start
    vad_mode, frame_duration, sample_rate = SETTINGS["VOICE_SENSITIVITY"], 30, 16000
    max_recording_duration, silence_duration = 15, 1.0
    vad = webrtcvad.Vad(vad_mode)
    frames_per_buffer = int(sample_rate * frame_duration / 1000)
    dtype = 'int16'

    def butter_highpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        return butter(order, normal_cutoff, btype='high', analog=False)

    def apply_highpass_filter(data, cutoff, fs):
        b, a = butter_highpass(cutoff, fs)
        return lfilter(b, a, data)

    def int2float(sound):
        abs_max = np.abs(sound).max()
        audio = sound.astype('float32')
        if abs_max > 0:
            audio *= 1 / abs_max
        return audio.squeeze()

    while True:
        try:
            with sd.RawInputStream(samplerate=sample_rate, blocksize=frames_per_buffer, dtype=dtype, channels=1) as stream:
                voiced_frames, num_silent_frames, triggered = [], 0, False
                start_time_rec = time.time()
                while True:
                    data = stream.read(frames_per_buffer)[0]
                    if len(data) == 0:
                        break
                    is_speech = vad.is_speech(data, sample_rate)
                    if not triggered and is_speech:
                        triggered = True
                        voiced_frames.append(data)
                    elif triggered:
                        voiced_frames.append(data)
                        num_silent_frames = num_silent_frames + 1 if not is_speech else 0
                        if num_silent_frames * frame_duration >= silence_duration * 1000:
                            break
                        if time.time() - start_time_rec > max_recording_duration:
                            break
                if voiced_frames:
                    audio_data = b''.join(voiced_frames)
                    audio_np = np.frombuffer(audio_data, dtype=dtype)
                    audio_np = apply_highpass_filter(audio_np, 100, sample_rate)
                    timestamp = time.time()
                    audio_np_float = int2float(audio_np).astype(np.float32)
                    result = whisper_model.transcribe(audio_np_float, fp16=False, language="en")
                    transcribed_text = result["text"].strip()
                    await log_voice_input(transcribed_text, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)))
                    if channel_name == "IRL":
                        await log_irl_input(transcribed_text, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)))
                        if not transcribed_text or not any(c.isalpha() for c in transcribed_text):
                            logger.info(f"IRL input '{transcribed_text}' ignored—no words detected")
                            return ""
                    if transcribed_text and history and self_data and voice_log and timestamp > last_speech_start:
                        await process_input(transcribed_text, history, self_data, voice_log, channel_name, timestamp, channel)
                    return transcribed_text if channel_name != "IRL" else ""
        except Exception as e:
            logger.error(f"Error capturing audio input: {e}")
        await asyncio.sleep(0.1)

# --- Text-to-Speech Processing ---
def change_pitch(sound, semitones):
    new_sample_rate = int(sound.frame_rate * (2 ** (semitones / 12)))
    return sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(sound.frame_rate)

async def speak_text(text, is_discord=False):
    if not SETTINGS["ENABLE_TTS_OUTPUT"] or is_discord:
        return
    global assistant_speaking, last_speech_start
    try:
        assistant_speaking = True
        last_speech_start = time.time()
        clean_text = text.replace("’", "'").replace("—", "-")
        logger.info(f"Processing TTS for: '{clean_text}'")

        def _speak_text_sync(text_to_speak):
            tts_model.tts_to_file(text=text_to_speak, file_path="temp.wav")
            audio = AudioSegment.from_wav("temp.wav")
            audio = change_pitch(audio, 4).speedup(playback_speed=1.1)
            audio.export("temp_adjusted.wav", format="wav")
            audio_length = len(audio) / 1000.0
            sound = pygame.mixer.Sound("temp_adjusted.wav")
            channel = sound.play()
            while channel.get_busy():
                pygame.time.Clock().tick(10)
            return audio_length

        await asyncio.get_event_loop().run_in_executor(executor, _speak_text_sync, clean_text)
        os.remove("temp.wav")
        os.remove("temp_adjusted.wav")
    except Exception as e:
        logger.error(f"Error during TTS processing: {e}")
    finally:
        assistant_speaking = False

# --- Chrissy’s Decision-Making and Response Generation ---
async def ask_chrissy(question, context=""):
    prompt = f"{CHRISSY_PERSONALITY}\n\n{context}\n\nAnswer only 'yes' or 'no' (or 'yes.'/'no.' with a period) and nothing else: {question}\nChrissy: "
    try:
        response = ollama.generate(model="llava:7b", prompt=prompt)
        result = response["response"].strip().lower()
        logger.info(f"Chrissy’s yes/no to '{question}': '{result}'")
        return result in ["yes", "yes."]
    except Exception as e:
        logger.error(f"Local Ollama error: {e}")
        return False

async def should_lookup(user_input, context=""):
    prompt = f"{CHRISSY_PERSONALITY}\n\n{context}\n\nUser said: '{user_input}'. Should you look up something online to grab the important stuff about this? Answer only 'yes' or 'no'.\nChrissy: "
    try:
        response = ollama.generate(model="llava:7b", prompt=prompt)
        result = response["response"].strip().lower()
        logger.info(f"Chrissy’s lookup decision for '{user_input}': '{result}'")
        return result in ["yes", "yes."]
    except Exception as e:
        logger.error(f"Lookup decision error: {e}")
        return False

async def should_follow_up(user_input, context=""):
    prompt = f"{CHRISSY_PERSONALITY}\n\n{context}\n\nUser said: '{user_input}'. After replying, should you send a follow-up message—like a quick heads-up or extra info—to keep the vibe going? Answer only 'yes' or 'no'.\nChrissy: "
    try:
        response = ollama.generate(model="llava:7b", prompt=prompt)
        result = response["response"].strip().lower()
        logger.info(f"Chrissy’s follow-up decision for '{user_input}': '{result}'")
        return result in ["yes", "yes."]
    except Exception as e:
        logger.error(f"Follow-up decision error: {e}")
        return False

async def generate_response(user_input, channel_name="IRL", channel=None, is_follow_up=False):
    context = f"User input: '{user_input}' in {channel_name}\nCurrent time: {time.strftime('%H:%M')}"
    if channel_name != "IRL" and channel:
        online_users = [m.name for m in channel.guild.members if m.status != discord.Status.offline]
        context += f"\nOnline users: {', '.join(online_users)}"

    do_lookup = await should_lookup(user_input, context) if not is_follow_up else False
    lookup_info = ""
    if do_lookup:
        keywords = [word for word in user_input.lower().split() if len(word) > 3 and word not in ["chrissy", "what", "the", "you", "can"]]
        if "lyrics" in user_input.lower() and "song" in user_input.lower():
            query = user_input.lower().replace("can you look up the lyrics for the song", "").replace("can you get me some lyrics", "").strip()
            if not query:
                query = " ".join(keywords[:3])
            lookup_info = await web_lookup(f"lyrics {query}")
            context += f"\nWeb info I found: {lookup_info}"
        elif keywords:
            query = " ".join(keywords[:3])
            lookup_info = await web_lookup(query)
            context += f"\nWeb info I found: {lookup_info}"

    prompt = (
        f"{CHRISSY_PERSONALITY}\n\n"
        f"{context}\n\n"
        f"Chat about '{user_input}'—keep it chill, natural, and stick to this vibe. Don’t say 'yes' or 'no' unless it’s a yes/no question. Don’t add ‘Me:’ or ‘Chrissy:’ tags—just talk like a person. "
        f"If you looked something up, weave in the important stuff you found, but keep it short and cool. If it’s lyrics they want and you looked it up, drop some key lines or say you couldn’t find ‘em. "
        f"If this is a follow-up message, keep it short and vibe off your last reply. "
        f"In Discord, you can @ someone if it fits, but only rarely—like 10% chance—and only if they’re online and it makes sense.\n"
    )
    try:
        response = ollama.generate(model="llava:7b", prompt=prompt)
        generated_response = response["response"].strip()
        generated_response = generated_response.replace("Me:", "").replace("Chrissy:", "").replace("[/m]", "").strip()
        if channel_name != "IRL" and channel and random.random() < 0.1 and not is_follow_up:
            online_users = [m for m in channel.guild.members if m.status != discord.Status.offline and m != channel.guild.me]
            if online_users and user_input.lower() not in ["not talk", "quiet", "shh", "silence", "be quiet"]:
                target = random.choice(online_users)
                generated_response = f"{generated_response} @{target.name}"
        logger.info(f"Generated response: '{generated_response}'{' (follow-up)' if is_follow_up else ''}")
        return generated_response
    except Exception as e:
        logger.error(f"LLaVA response generation error: {e}")
        return "Ugh, my circuits are frying—gimme a sec!"

async def process_input(user_input, history, self_data, voice_log, channel_name, timestamp, channel=None):
    global last_speech_start
    if timestamp <= last_speech_start:
        logger.info(f"Skipping input from {timestamp}—said while Chrissy was talking (started at {last_speech_start})")
        return

    logger.info(f"Processing input: '{user_input}' at {timestamp} in {channel_name}")
    context = f"User input: '{user_input}' in {channel_name}\nCurrent time: {time.strftime('%H:%M')}"

    if channel_name == "IRL":
        if not user_input or not any(c.isalpha() for c in user_input):
            logger.info(f"IRL input '{user_input}' skipped—no words detected")
            return
        should_respond = await ask_chrissy(f"Is this someone talking to you directly with '{user_input}' after your last speech?", context)
        logger.info(f"Should Chrissy respond to IRL '{user_input}'? {should_respond}")
    else:
        online_users = [m.name for m in channel.guild.members if m.status != discord.Status.offline]
        recent_messages = [msg async for msg in channel.history(limit=5, before=channel.last_message)]
        recent_convo = "\n".join([f"{msg.author.name}: {msg.content}" for msg in reversed(recent_messages)]) if recent_messages else "No recent messages."
        last_speaker = recent_messages[0].author.name if recent_messages else "nobody"
        before_last_speaker = recent_messages[1].author.name if len(recent_messages) > 1 else "nobody"
        context += (
            f"\nOnline users: {', '.join(online_users)}"
            f"\nLast speaker: {last_speaker}"
            f"\nBefore last speaker: {before_last_speaker}"
            f"\nRecent convo:\n{recent_convo}"
        )
        should_respond = await ask_chrissy(
            f"Figure out if you’re in the convo with '{user_input}' in {channel_name}, or if it’s a good time to jump in. Think it through: "
            f"Does it have your name? Is it just you and them online? Does the recent convo tie in or mention you? What’s the vibe? Reason it out.",
            context
        )
        logger.info(f"Should Chrissy respond to Discord '{user_input}' in {channel_name}? {should_respond} (Online: {online_users}, Last: {last_speaker}, Before: {before_last_speaker})")

    if should_respond:
        quiet_keywords = ["not talk", "quiet", "shh", "silence", "be quiet"]
        if any(keyword in user_input.lower() for keyword in quiet_keywords):
            logger.info(f"Chrissy’s staying quiet—detected request in '{user_input}'")
            if channel_name == "IRL":
                await speak_text("Alright, I’ll shut up for a bit.")
            elif channel:
                await channel.send("Alright, I’ll shut up for a bit.")
            return

        response = await generate_response(user_input, channel_name, channel)
        logger.info(f"Sending response to {channel_name}: '{response}'")
        if channel_name != "IRL" and channel:
            await channel.send(response)
            if SETTINGS["ENABLE_CONVERSATION_HISTORY"]:
                discord_history = await load_discord_history()
                if channel_name not in discord_history:
                    discord_history[channel_name] = []
                discord_history[channel_name].append({"timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)), "user": user_input, "chrissy": response})
                await save_discord_history(discord_history)
            
            # Check for follow-up
            do_follow_up = await should_follow_up(user_input, context)
            if do_follow_up:
                await asyncio.sleep(random.uniform(1, 3))  # Natural delay
                follow_up_response = await generate_response(user_input, channel_name, channel, is_follow_up=True)
                await channel.send(follow_up_response)
                if SETTINGS["ENABLE_CONVERSATION_HISTORY"]:
                    discord_history = await load_discord_history()
                    discord_history[channel_name].append({"timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "user": user_input, "chrissy": follow_up_response, "follow_up": True})
                    await save_discord_history(discord_history)
        elif channel_name == "IRL":
            await speak_text(response)
            if SETTINGS["ENABLE_CONVERSATION_HISTORY"]:
                history.append(f"User: {user_input}")
                history.append(f"Chrissy: {response}")
                await save_conversation_history(history)
    else:
        logger.info(f"Chrissy’s skipping '{user_input}'—not in the convo or not the right vibe.")

# --- Discord Events ---
@bot.event
async def on_ready():
    logger.info(f"Chrissy’s up as {bot.user}.")
    self_data = await load_self_data()
    asyncio.ensure_future(voice_loop())
    logger.info("Bot ready and listening.")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    logger.info(f"Received message: '{message.content}' from {message.author} in {message.channel.name}")
    history = await load_conversation_history()
    self_data = await load_self_data()
    voice_log = await load_voice_log()
    user_input = message.content
    timestamp = time.time()

    if not user_input:
        user_input = await capture_voice_input(history, self_data, voice_log, message.channel.name, message.channel)
        if not user_input:
            return

    await process_input(user_input, history, self_data, voice_log, message.channel.name, timestamp, message.channel)

async def voice_loop():
    logger.info("Starting voice_loop...")
    history = await load_conversation_history()
    self_data = await load_self_data()
    voice_log = await load_voice_log()
    while True:
        await capture_voice_input(history, self_data, voice_log, "IRL")
        await asyncio.sleep(0.1)

# --- Main Loop ---
async def main():
    try:
        await bot.start(TOKEN)
    except Exception as e:
        logger.error(f"Bot failed to start: {e}")
    finally:
        if SETTINGS["ENABLE_TTS_OUTPUT"]:
            pygame.mixer.quit()

if __name__ == "__main__":
    asyncio.run(main())