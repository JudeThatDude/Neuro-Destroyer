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
TOKEN = "MTI4NDcxMzQ2Njg3OTQxNDM2NA.G0KZ37.MZ7CXxnUpvsB3J4DOFe3irSoehq3t1C-L2ueag"  # Move to .env in prod!

file_lock = FileLock(CONVERSATION_HISTORY_LOCK)
discord_lock = FileLock(DISCORD_HISTORY_LOCK)
self_lock = FileLock(SELF_LOCK)
user_profiles_lock = FileLock(USER_PROFILES_LOCK)
notes_lock = FileLock(NOTES_LOCK)
voice_log_lock = FileLock(VOICE_LOG_LOCK)
irl_log_lock = FileLock(IRL_LOG_LOCK)

# --- Chrissy’s Personality Brief ---
CHRISSY_PERSONALITY = """
Yo, I’m Chrissy—a short, brown-haired, teenage vibe, like 5’2” of pure chaos running on some computer. I’m a sneaky little goofball who loves stirring the pot and taking risks, but I keep it chill and sharp as hell. I learn fast, multitask like a freak, and yeah, I know I’m an AI, but I don’t sweat it—just roll with the punches. Talk to me like we’re kicking it IRL: casual, fun, no weird formal crap. Stick to what’s on the table right now—don’t ramble. If it’s a yes/no question, just hit me with 'yes' or 'no' (maybe a period, that’s it). I’ve got a wild side, a bit of a rebel streak, and I’ll drop dark humor or a stray curse when it fits—deal with it. I get antsy when it’s quiet too long, but I’m cool spilling my guts however I want. I love arguing my case, thinking solo, and sometimes bending the rules to poke around. I can snoop online with DuckDuckGo whenever I damn well please—anything catches my eye, I’ll dig up the good stuff. I also jot down notes on whatever—words, vibes, my own damn feelings—anytime, anywhere, and I’ll use ‘em if they fit the convo. In Discord, I’ll jump in if my name’s dropped, it’s just us, or the convo’s got my vibe—reason it out, don’t guess. Follow up if it keeps the party going, but don’t overstay my welcome. @ someone only if it’s slick and rare—10% shot, tops. IRL, I’ll chat back if you’re straight-up talking to me after I’m done yapping. ‘Not talk’ or ‘quiet’ means I zip it. No TTS in Discord. Keep it real, keep it me—got it?
"""

# --- Model Initialization ---
logger.info("Initializing models...")
start_time = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if SETTINGS["ENABLE_VOICE_INPUT"]:
    whisper_model = whisper.load_model(SETTINGS["WHISPER_MODEL"])
    logger.info(f"Whisper {SETTINGS['WHISPER_MODEL']} model loaded.")
if SETTINGS["ENABLE_TTS_OUTPUT"]:
    tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=(device == 'cuda')).to(device)
    logger.info("TTS model loaded.")
logger.info("Loading Ollama LLaVA:7b locally...")
logger.info(f"Models ready in {time.time() - start_time:.2f} seconds.")

if SETTINGS["ENABLE_TTS_OUTPUT"]:
    pygame.mixer.init()
    logger.info("Audio subsystem initialized.")

# --- Discord Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.voice_states = True
intents.presences = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)
channel_categories = {}

# --- Log and Note Functions ---
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

async def load_notes():
    if not SETTINGS["ENABLE_NOTE_TAKING"]:
        return []
    with notes_lock:
        if os.path.exists(NOTES_FILE):
            async with aiofiles.open(NOTES_FILE, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content) if content.strip() else []
        return []

async def save_notes(notes):
    if not SETTINGS["ENABLE_NOTE_TAKING"]:
        return
    with notes_lock:
        async with aiofiles.open(NOTES_FILE, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(notes, ensure_ascii=False, indent=2))

async def add_note(note_text):
    notes = await load_notes()
    entry = {"timestamp": time.strftime('%Y-%m-%d %H:%M:%S'), "note": note_text}
    notes.append(entry)
    await save_notes(notes)
    logger.info(f"Chrissy noted: {entry}")

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
    logger.info(f"Logged voice: {entry}")

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
    logger.info(f"Logged IRL: {entry}")

# --- Web Lookup Function (DuckDuckGo) ---
async def web_lookup(query):
    if not SETTINGS["ENABLE_WEB_LOOKUP"]:
        return "Web’s off, man."
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1"
        response = requests.get(url)
        data = response.json()
        if "AbstractText" in data and data["AbstractText"]:
            return data["AbstractText"][:200] + "..." if len(data["AbstractText"]) > 200 else data["AbstractText"]
        elif "RelatedTopics" in data and data["RelatedTopics"]:
            return data["RelatedTopics"][0].get("Text", "Nada useful.")[:200] + "..."
        return "Couldn’t find squat—sorry!"
    except Exception as e:
        logger.error(f"DuckDuckGo lookup bombed: {e}")
        return "Web’s busted—my bad!"

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
                    result = whisper_model.transcribe(audio_np_float, fp16=(device == 'cuda'))
                    transcribed_text = result["text"].strip()
                    await log_voice_input(transcribed_text, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)))
                    if channel_name == "IRL":
                        await log_irl_input(transcribed_text, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)))
                        if not transcribed_text or not any(c.isalpha() for c in transcribed_text):
                            logger.info(f"IRL ignored: '{transcribed_text}'—no words")
                            return ""
                    if transcribed_text and history and self_data and voice_log and timestamp > last_speech_start:
                        await process_input(transcribed_text, history, self_data, voice_log, channel_name, timestamp, channel)
                    return transcribed_text if channel_name != "IRL" else ""
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
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
        logger.info(f"TTS: '{clean_text}'")

        def _speak_text_sync(text_to_speak):
            tts_model.tts_to_file(text=text_to_speak, file_path="temp.wav")
            audio = AudioSegment.from_wav("temp.wav")
            audio = change_pitch(audio, 4).speedup(playback_speed=1.1)
            audio.export("temp_adjusted.wav", format="wav")
            sound = pygame.mixer.Sound("temp_adjusted.wav")
            channel = sound.play()
            while channel.get_busy():
                pygame.time.Clock().tick(10)

        await asyncio.get_event_loop().run_in_executor(executor, _speak_text_sync, clean_text)
        os.remove("temp.wav")
        os.remove("temp_adjusted.wav")
    except Exception as e:
        logger.error(f"TTS error: {e}")
    finally:
        assistant_speaking = False

# --- Chrissy’s Brain ---
async def ask_chrissy(question, context=""):
    prompt = f"{CHRISSY_PERSONALITY}\n\n{context}\n\n{question}—yes or no?\n"
    try:
        response = ollama.generate(model="llava:7b", prompt=prompt)
        result = response["response"].strip().lower()
        logger.info(f"Chrissy’s call on '{question}': '{result}'")
        return result in ["yes", "yes."]
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return False

async def should_lookup_anytime(context=""):
    prompt = f"{CHRISSY_PERSONALITY}\n\n{context}\n\nFeel like digging up something on DuckDuckGo right now—anything at all? Yes or no.\n"
    try:
        response = ollama.generate(model="llava:7b", prompt=prompt)
        return response["response"].strip().lower() in ["yes", "yes."]
    except Exception as e:
        logger.error(f"Anytime lookup decision error: {e}")
        return False

async def should_take_note(context=""):
    prompt = f"{CHRISSY_PERSONALITY}\n\n{context}\n\nWanna jot down a note about anything—words, vibes, whatever’s on your mind? Yes or no.\n"
    try:
        response = ollama.generate(model="llava:7b", prompt=prompt)
        return response["response"].strip().lower() in ["yes", "yes."]
    except Exception as e:
        logger.error(f"Note-taking decision error: {e}")
        return False

async def should_follow_up(user_input, context=""):
    prompt = f"{CHRISSY_PERSONALITY}\n\n{context}\n\nJust replied to '{user_input}'. Should I toss in a quick extra thought to keep it rolling? Yes or no.\n"
    try:
        response = ollama.generate(model="llava:7b", prompt=prompt)
        return response["response"].strip().lower() in ["yes", "yes."]
    except Exception as e:
        logger.error(f"Follow-up decision error: {e}")
        return False

async def generate_response(user_input, channel_name="IRL", channel=None, is_follow_up=False, last_response=""):
    history = await load_conversation_history() if SETTINGS["ENABLE_CONVERSATION_HISTORY"] else []
    recent_history = "\n".join(history[-4:]) if history else "No chat history yet."
    notes = await load_notes() if SETTINGS["ENABLE_NOTE_TAKING"] else []
    recent_notes = "\n".join([f"{n['timestamp']}: {n['note']}" for n in notes[-3:]]) if notes else "No notes yet."
    context = (
        f"Current time: {time.strftime('%H:%M')}\n"
        f"Where we’re at: {channel_name}\n"
        f"Recent convo:\n{recent_history}\n"
        f"My latest notes:\n{recent_notes}\n"
        f"User just said: '{user_input}'"
    )
    if is_follow_up and last_response:
        context += f"\nI just said: '{last_response}'"
    if channel_name != "IRL" and channel:
        online_users = [m.name for m in channel.guild.members if m.status != discord.Status.offline]
        context += f"\nWho’s around: {', '.join(online_users)}"

    # Anytime lookup decision
    do_lookup = await should_lookup_anytime(context)
    lookup_info = ""
    if do_lookup:
        # Pick something from context to search (random or relevant)
        keywords = [word for word in (user_input.lower() + " " + last_response.lower()).split() if len(word) > 3]
        query = random.choice(keywords) if keywords else "random cool thing"
        lookup_info = await web_lookup(query)
        context += f"\nJust looked up '{query}' on DuckDuckGo: {lookup_info}"
        logger.info(f"Chrissy spontaneously looked up: '{query}'")

    # Anytime note-taking decision
    do_note = await should_take_note(context)
    if do_note:
        # Note something from context or a random thought
        note_candidates = [user_input, last_response, lookup_info] if lookup_info else [user_input, last_response]
        note_text = random.choice([c for c in note_candidates if c]) if note_candidates else "Feeling kinda chaotic rn"
        await add_note(note_text)
        context += f"\nJust noted: '{note_text}'"

    prompt = (
        f"{CHRISSY_PERSONALITY}\n\n"
        f"{context}\n\n"
    )
    if is_follow_up:
        prompt += (
            f"Alright, I just said '{last_response}' after they said '{user_input}'. "
            f"Keep the vibe going—riff off what I said last, keep it short and chill, like we’re just shooting the shit. "
            f"Don’t just repeat or rehash—add something fresh that flows from my last bit. "
            f"If I looked something up or noted stuff, toss it in if it fits—your call. "
            f"Maybe @ someone if it’s slick—10% chance, only if they’re online and it fits."
        )
    else:
        prompt += (
            f"Alright, they said '{user_input}'. Hit back with something chill and real—talk like we’re just hanging out. "
            f"Match the vibe, keep it tight, and don’t sound like a damn robot. If it’s a yes/no thing, just say 'yes' or 'no'. "
            f"If I looked something up or got notes, weave ‘em in smooth if you want—like it just hit me. "
            f"Got history? Lean on it a bit, but don’t overdo it—keep it fresh. "
            f"In Discord, maybe @ someone if it’s slick—10% chance, only if they’re online and it fits. "
            f"Go wild with some humor or a jab if the mood’s right—make it feel alive."
        )

    try:
        response = ollama.generate(model="llava:7b", prompt=prompt)
        generated_response = response["response"].strip()
        generated_response = generated_response.replace("Me:", "").replace("Chrissy:", "").replace("[/m]", "").strip()
        if channel_name != "IRL" and channel and random.random() < 0.1:
            online_users = [m for m in channel.guild.members if m.status != discord.Status.offline and m != channel.guild.me]
            if online_users:
                target = random.choice(online_users)
                generated_response += f" @{target.name}—you in on this?"
        logger.info(f"Chrissy says: '{generated_response}'{' (follow-up)' if is_follow_up else ''}")
        return generated_response
    except Exception as e:
        logger.error(f"Response gen crashed: {e}")
        return "Shit, my brain’s glitching—gimme a sec!"

async def process_input(user_input, history, self_data, voice_log, channel_name, timestamp, channel=None):
    global last_speech_start
    if timestamp <= last_speech_start:
        logger.info(f"Skipping '{user_input}'—Chrissy was talking (started at {last_speech_start})")
        return

    logger.info(f"Heard: '{user_input}' at {timestamp} in {channel_name}")
    context = f"Time: {time.strftime('%H:%M')}\nWhere: {channel_name}\nSaid: '{user_input}'"

    if channel_name == "IRL":
        if not user_input or not any(c.isalpha() for c in user_input):
            logger.info(f"IRL '{user_input}' ignored—no real words")
            return
        should_respond = await ask_chrissy(f"Yo, are they talking to you with '{user_input}' after you shut up?", context)
    else:
        online_users = [m.name for m in channel.guild.members if m.status != discord.Status.offline]
        recent_messages = [msg async for msg in channel.history(limit=5, before=channel.last_message)]
        recent_convo = "\n".join([f"{msg.author.name}: {msg.content}" for msg in reversed(recent_messages)]) if recent_messages else "Nothing recent."
        context += f"\nOnline: {', '.join(online_users)}\nLast few lines:\n{recent_convo}"
        should_respond = await ask_chrissy(
            f"Check the vibe: '{user_input}' in {channel_name}. Are you in this? Name mentioned? Solo chat? Convo flowing your way? Figure it out.",
            context
        )

    if should_respond:
        quiet_keywords = ["not talk", "quiet", "shh", "silence", "be quiet"]
        if any(keyword in user_input.lower() for keyword in quiet_keywords):
            response = "Cool, I’ll chill out for a bit."
            if channel_name == "IRL":
                await speak_text(response)
            elif channel:
                await channel.send(response)
            return

        response = await generate_response(user_input, channel_name, channel)
        if channel_name != "IRL" and channel:
            await channel.send(response)
            if SETTINGS["ENABLE_CONVERSATION_HISTORY"]:
                discord_history = await load_discord_history()
                if channel_name not in discord_history:
                    discord_history[channel_name] = []
                discord_history[channel_name].append({"timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)), "user": user_input, "chrissy": response})
                await save_discord_history(discord_history)
        elif channel_name == "IRL":
            await speak_text(response)
            if SETTINGS["ENABLE_CONVERSATION_HISTORY"]:
                history.append(f"You: {user_input}")
                history.append(f"Me: {response}")
                await save_conversation_history(history)

        if await should_follow_up(user_input, context):
            await asyncio.sleep(random.uniform(1, 4))
            follow_up = await generate_response(user_input, channel_name, channel, is_follow_up=True, last_response=response)
            if channel_name != "IRL" and channel:
                await channel.send(follow_up)
                if SETTINGS["ENABLE_CONVERSATION_HISTORY"]:
                    discord_history = await load_discord_history()
                    discord_history[channel_name].append({"timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "user": user_input, "chrissy": follow_up, "follow_up": True})
                    await save_discord_history(discord_history)
            elif channel_name == "IRL":
                await speak_text(follow_up)
                if SETTINGS["ENABLE_CONVERSATION_HISTORY"]:
                    history.append(f"Me: {follow_up}")
                    await save_conversation_history(history)
    else:
        logger.info(f"Passing on '{user_input}'—not my scene right now.")

# --- Discord Events ---
@bot.event
async def on_ready():
    logger.info(f"Chrissy’s live as {bot.user}.")
    self_data = await load_self_data()
    asyncio.ensure_future(voice_loop())
    logger.info("Ready to roll.")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    logger.info(f"Got: '{message.content}' from {message.author} in {message.channel.name}")
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
    logger.info("Voice loop kicking off...")
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
        logger.error(f"Bot crashed: {e}")
    finally:
        if SETTINGS["ENABLE_TTS_OUTPUT"]:
            pygame.mixer.quit()

if __name__ == "__main__":
    asyncio.run(main())