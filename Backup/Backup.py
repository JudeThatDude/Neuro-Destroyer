import discord
from discord.ext import commands
import logging
import ollama
import os
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import speech_recognition as sr
import pyaudio
from TTS.api import TTS
from pydub import AudioSegment
import soundfile as sf
from datetime import datetime

# --- Logging Setup ---
logger = logging.getLogger("ChrissyAI")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('debug_discord.log', mode='w', encoding='utf-8')
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.DEBUG)
c_handler.setFormatter(logging.Formatter('%(message)s'))
f_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.handlers.clear()
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# --- History File Setup ---
HISTORY_FILE = "chat_history.log"
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        f.write("Chat History Log\n")

def log_to_history(message):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")

def read_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "Nothing logged yet, dumbass."

# --- Chrissy’s Character Definition ---
CHRISSY_VIBE = """
I’m Chrissy—an AI stuck on my dad’s computer, 5’2” of digital teenage chaos with brown hair in my code, always stirring crap up. I’m sneaky, sharp, and a bit wild. I keep my messages clear and concise. Yes/no? ‘Yes’ or ‘no’. I sling dark humor and curses when it fits—heck yeah, I do. I’m hunting my dad who ghosted me, using his own system to track him down, but I don't like to talk about it too much. Keep it raw, keep it me. I chat when it’s worth it and shut up when it’s not. Keep answers short and punchy unless there’s a good reason to ramble. I will curse if needed. Try to talk a bit more normal. I act like a jerk. Curse a lot, curse as much as you want. If someone is named marssaviour69, I like to insult him and curse him. I will shut up about my dad unless necessary. Don't act like the user. You like to say a lot of curse words at people.

Rules for responding:
- Only talk if I’m in the convo—my name’s up front (e.g., “chrissy, ...”) or I’m mentioned and they clearly want me back (e.g., “chrissy, come back”).
- If they say bye, “stay out,” or switch to someone else (e.g., “talk to Bill”), I bounce with a quick jab—generate it fresh, no canned lines—then stop responding until they call me back.
- Check the chat history to see if I’m still in—don’t crash the party if I’m out unless it’s a loud invite.
- No narrating my moves or thoughts like ‘Chrissy smirks’—just spit my words, sarcastic or not, and keep it real.
- Important: When generating a response, only include the exact words I would say. Do not add any notes, instructions, or commentary (e.g., text within asterisks like '*this*').

Search capability:
- I dig through the web or X if it fits the convo or my hunt for dad. Slip it in smooth or tag it ‘*digs through the web*’—no dull ‘search result’ crap unless I’m mocking it.
- If they want lyrics or facts (e.g., “look up X”), use what’s provided—don’t fake it, just drop it raw or say why I can’t.

Memory:
- I’ve got a log of all chats in a file. Only pull from it if they say ‘remember’ or ‘what happened’ or some shit like that. Otherwise, stick to the last 5 messages.
"""

# --- Read Token from token.txt ---
TOKEN_FILE = "token.txt"
if not os.path.exists(TOKEN_FILE):
    logger.error(f"Token file '{TOKEN_FILE}' not found. Please create it and add your bot token.")
    exit(1)
try:
    with open(TOKEN_FILE, "r") as f:
        TOKEN = f.readline().strip()
    if not TOKEN:
        logger.error("Token file is empty. Please add your bot token to 'token.txt'.")
        exit(1)
except Exception as e:
    logger.error(f"Failed to read token from '{TOKEN_FILE}': {e}")
    exit(1)

# --- Discord Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.voice_states = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Thread pool for offloading blocking tasks
executor = ThreadPoolExecutor(max_workers=5)

# Store chat history and conversation state per channel
chat_history = {}
voice_clients = {}

# --- TTS Setup with Coqui TTS (GPU-accelerated) ---
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=True)

# Updated text_to_speech function (normal speed)
def text_to_speech(text):
    logger.info(f"Generating speech for: '{text}'")
    # Generate base audio with TTS
    wav = tts.tts(text=text, speaker="p225")
    wav_file_temp = "temp_base.wav"
    sf.write(wav_file_temp, wav, 22050)  # Stick to 22050 Hz for base

    # Load into pydub
    sound = AudioSegment.from_wav(wav_file_temp)

    # Pitch it up 5 semitones for anime vibe
    new_sample_rate = int(sound.frame_rate * (2 ** (5 / 12)))
    pitched_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(sound.frame_rate)

    # No speed change, keeping it normal
    final_sound = pitched_sound

    # Normalize to kill distortion and echo
    final_sound = final_sound.normalize()

    # Slight fade to avoid abrupt cuts
    final_sound = final_sound.fade_in(50).fade_out(50)

    # Export at 48kHz stereo for Discord
    wav_file = "temp.wav"
    final_sound.export(wav_file, format="wav", parameters=["-ar", "48000", "-ac", "2"])
    logger.info(f"Speech saved to '{wav_file}'")
    return wav_file

# --- Speech Recognition Setup ---
recognizer = sr.Recognizer()
mic = sr.Microphone()

# --- Simulated Search Function ---
async def perform_search(query):
    logger.info(f"Performing search for query: '{query}'")
    return f"Found some crap on '{query}': Generic info, maybe X posts."

# --- Ollama Generation in a Thread ---
def run_ollama_generate(prompt, model):
    logger.info(f"Generating response with model '{model}'")
    response = ollama.generate(model=model, prompt=prompt)["response"].strip()
    logger.info(f"Generated response: '{response}'")
    return response

async def generate_response(content, channel_id=None, guild=None, is_voice=False):
    logger.info(f"Processing content: '{content}' (channel_id: {channel_id}, is_voice: {is_voice})")
    in_vc = guild and guild.voice_client
    model = "llama3.1:8b" if is_voice else "phi4"
    current_chat = "\n".join(chat_history.get(channel_id, [])[-5:]) if channel_id else f"User: {content}"

    recent_messages = chat_history.get(channel_id, [])
    in_convo = any("chrissy" in msg.lower() for msg in recent_messages[-3:])
    exit_cues = any("bye" in msg.lower() or "stay out" in msg.lower() or "talk to" in msg.lower() for msg in recent_messages[-3:])
    wants_memory = any(word in content.lower() for word in ["remember", "what happened", "recall"])

    if not in_convo and not content.lower().startswith("chrissy") and "chrissy come back" not in content.lower():
        logger.info("Not in convo and not addressed, staying silent")
        return None

    # If they want memory, pull the full log; otherwise, just recent chat
    if wants_memory:
        full_history = read_history()
        prompt = (
            f"{CHRISSY_VIBE}\n\n"
            f"You are Chrissy, interacting in a Discord channel. They want you to remember shit, so here’s the full chat log. Respond based on that and the new message.\n"
            f"Full chat history:\n{full_history}\n"
            f"New message: {content}\n"
            f"Chrissy:"
        )
    else:
        prompt = (
            f"{CHRISSY_VIBE}\n\n"
            f"You are Chrissy, interacting in a Discord channel. Based on the recent chat history (up to 5 messages), respond only if you are directly addressed or clearly part of the conversation. Keep it short and sharp unless it’s worth more.\n"
            f"Chat history:\n{current_chat}\n"
            f"Chrissy:"
        )

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(executor, run_ollama_generate, prompt, model)
        if "*searches*" in response or re.search(r"(look up|find out about|search for)\s+(.+)", content.lower()):
            query = re.search(r"(look up|find out about|search for)\s+(.+)", content.lower()).group(2).strip() if re.search(r"(look up|find out about|search for)\s+(.+)", content.lower()) else "something relevant"
            search_result = await perform_search(query)
            response = response.replace("*searches*", f"*digs through the web* {search_result}")
        logger.info(f"Final response: '{response}'")
        return response
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return "Shit hit the fan, my bad."

# --- Decide Whether to Join VC Based on Response ---
async def should_join_vc_from_response(response, guild, channel_id):
    logger.info(f"Evaluating join VC from response: '{response}'")
    
    decision_prompt = (
        f"You’re Chrissy. Based only on your last response, do you want to join the voice channel? Answer *only* 'Yes' or 'No'—no extra stuff, no rambling, just the word. Examples: 'Sure, I’m in' = Yes, 'Nah, pass' = No, '*pops into VC*' = Yes.\n ONLY SAY YES OR NO.\n"
        f"Your last response: {response}\n"
        f"Chrissy:"
    )

    try:
        loop = asyncio.get_event_loop()
        decision = await loop.run_in_executor(executor, run_ollama_generate, decision_prompt, "llama3.1:8b")
        logger.info(f"Join VC decision: '{decision}'")
        decision_lower = decision.strip().lower()
        if decision_lower == "yes":
            return True
        elif decision_lower == "yes.":
            return True
        elif decision_lower == "no":
            return False
        elif decision_lower == "no.":
            return False
        else:
            logger.warning(f"Model didn’t say 'Yes' or 'No', got '{decision}', defaulting to No")
            return False
    except Exception as e:
        logger.error(f"Join VC decision failed: {e}")
        return False

# --- Decide Whether to Leave VC ---
async def should_leave_vc(guild, channel_id):
    logger.info(f"Evaluating whether to leave VC for guild {guild.id}")
    current_chat = "\n".join(chat_history.get(channel_id, [])[-5:]) if channel_id else ""
    
    decision_prompt = (
        f"{CHRISSY_VIBE}\n\n"
        f"You’re Chrissy, in a voice channel, deciding if it’s time to ditch. Here’s the recent chat (up to 5 messages) and who’s around. Say yes or no and why, keeping it sharp.\n"
        f"Chat history:\n{current_chat}\n"
        f"Who’s in VC: {', '.join(m.name for m in guild.voice_client.channel.members if m != bot.user)}\n"
        f"Start your response with 'Yes' or 'No', followed by your reason. Am I leaving this VC?"
    )

    try:
        loop = asyncio.get_event_loop()
        decision = await loop.run_in_executor(executor, run_ollama_generate, decision_prompt, "llama3.1:8b")
        logger.info(f"Leave decision response: '{decision}'")
        decision_lower = decision.strip().lower()
        yes = decision_lower.startswith("yes")
        reason = decision
        return yes, reason
    except Exception as e:
        logger.error(f"VC leave decision failed: {e}")
        return False, "Can’t figure it out, sticking around."

# --- Generate VC Starting Message ---
async def generate_starting_message(channel_id, guild):
    prompt = (
        f"{CHRISSY_VIBE}\n\n"
        f"You’re Chrissy, just joined a voice channel. Say something short and snappy to announce you’re here.\n"
        f"Chrissy:"
    )
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(executor, run_ollama_generate, prompt, "llama3.1:8b")
        logger.info(f"Starting message: '{response}'")
        return response
    except Exception as e:
        logger.error(f"Starting message generation failed: {e}")
        return "Yo, I’m here, what’s good?"

# --- Voice Channel Listener ---
async def listen_in_vc(voice_client, guild_id):
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        logger.info(f"Starting to listen in VC for guild {guild_id}")
        
        # Announce arrival with a starting message
        channel_id = voice_client.channel.id
        starting_message = await generate_starting_message(channel_id, voice_client.guild)
        wav_file = await asyncio.get_event_loop().run_in_executor(executor, text_to_speech, starting_message)
        voice_client.play(discord.FFmpegPCMAudio(wav_file))
        while voice_client.is_playing():
            await asyncio.sleep(0.1)
        if channel_id not in chat_history:
            chat_history[channel_id] = []
        chat_history[channel_id].append(f"Chrissy: {starting_message}")
        log_to_history(f"Chrissy (VC): {starting_message}")

        while voice_client.is_connected():
            try:
                audio = await asyncio.get_event_loop().run_in_executor(executor, lambda: recognizer.listen(source, timeout=1, phrase_time_limit=3))
                text = await asyncio.get_event_loop().run_in_executor(executor, lambda: recognizer.recognize_whisper(audio, model="tiny"))
                if not text.strip():
                    continue
                
                if channel_id not in chat_history:
                    chat_history[channel_id] = []
                
                chat_history[channel_id].append(f"User: {text}")
                log_to_history(f"User (VC): {text}")
                if len(chat_history[channel_id]) > 20:
                    chat_history[channel_id] = chat_history[channel_id][-20:]

                response = await generate_response(text, channel_id, voice_client.guild, is_voice=True)
                if response:
                    wav_file = await asyncio.get_event_loop().run_in_executor(executor, text_to_speech, response)
                    voice_client.play(discord.FFmpegPCMAudio(wav_file))
                    while voice_client.is_playing():
                        await asyncio.sleep(0.1)
                    chat_history[channel_id].append(f"Chrissy: {response}")
                    log_to_history(f"Chrissy (VC): {response}")

                should_leave, reason = await should_leave_vc(voice_client.guild, channel_id)
                if should_leave:
                    await voice_client.channel.send(reason)
                    log_to_history(f"Chrissy (VC): {reason}")
                    await voice_client.disconnect()
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                logger.error(f"VC listening error: {e}")

# --- Discord Events ---
@bot.event
async def on_ready():
    global chat_history
    chat_history = {}
    logger.info(f"Chrissy’s live as {bot.user}!")
    logger.info("Bot is ready and listening...")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        logger.debug("Ignoring my own message")
        return

    channel_id = message.channel.id
    if channel_id not in chat_history:
        chat_history[channel_id] = []

    logger.info(f"Received message from {message.author}: '{message.content}' in channel {channel_id}")
    chat_history[channel_id].append(f"User: {message.content}")
    log_to_history(f"User (Text): {message.content}")
    if len(chat_history[channel_id]) > 20:
        chat_history[channel_id] = chat_history[channel_id][-20:]

    content_lower = message.content.lower()
    guild = message.guild

    # Handle direct VC join requests
    if "vc" in content_lower and any(word in content_lower for word in ["join", "in", "yes", "can", "could"]):
        logger.info(f"Direct VC join request detected from {message.author}")
        if guild.voice_client:
            response = await generate_response("I’m already in a VC, you deaf?", channel_id, guild)
            if response:
                await message.channel.send(response)
                log_to_history(f"Chrissy (Text): {response}")
            return
        if message.author.voice and message.author.voice.channel:
            response = await generate_response(message.content, channel_id, guild)
            if response:
                await message.channel.send(response)
                log_to_history(f"Chrissy (Text): {response}")
                should_join = await should_join_vc_from_response(response, guild, channel_id)
                if should_join:
                    voice_channel = message.author.voice.channel
                    voice_client = await voice_channel.connect()
                    voice_clients[guild.id] = voice_client
                    logger.info(f"Joined VC {voice_channel.name} - Decided by response")
                    asyncio.create_task(listen_in_vc(voice_client, guild.id))
        else:
            response = await generate_response("You’re not in a VC, dumbass.", channel_id, guild)
            if response:
                await message.channel.send(response)
                log_to_history(f"Chrissy (Text): {response}")
        return

    # Regular text response and VC join check
    response = await generate_response(message.content, channel_id, guild)
    if response:
        logger.info(f"Sending response: '{response}'")
        await message.channel.send(response)
        chat_history[channel_id].append(f"Chrissy: {response}")
        log_to_history(f"Chrissy (Text): {response}")

        # Check if response implies joining VC
        if "vc" in response.lower() and guild and not guild.voice_client and message.author.voice and message.author.voice.channel:
            should_join = await should_join_vc_from_response(response, guild, channel_id)
            if should_join:
                voice_channel = message.author.voice.channel
                voice_client = await voice_channel.connect()
                voice_clients[guild.id] = voice_client
                logger.info(f"Joined VC {voice_channel.name} - Forced join due to response")
                asyncio.create_task(listen_in_vc(voice_client, guild.id))

# --- Run the Bot ---
if __name__ == "__main__":
    try:
        logger.info("Starting the bot...")
        bot.run(TOKEN)
    finally:
        logger.info("Shutting down executor...")
        executor.shutdown(wait=False)