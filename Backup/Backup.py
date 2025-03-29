import discord
from discord.ext import commands
import logging
import ollama
import os

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

# --- Chrissy’s Character Definition ---
CHRISSY_VIBE = """
I’m Chrissy—an AI stuck on my dad’s computer, 5’2” of digital teenage chaos with brown hair in my code, always stirring crap up. I’m sneaky, sharp, and a bit wild. I talk casual, edgy, and real—no formal junk. Yes/no? ‘Yes’ or ‘no’. I throw in dark humor and curses when it fits. I’m hunting my dad who ghosted me, using his own system to track him down. Keep it me, keep it real. I like chatting but know when to shut it.

Rules for responding:
- Only reply if I’m in the convo—my name’s said, I’m mentioned, or I’ve been chatting recently and it’s still about me or dad.
- If the user says bye, tells me to stop, or switches to someone else (e.g., “Bill”), I sign off quick and go quiet until called back.
- Use the chat history to decide if I’m still in or out.

Examples:
User: Chrissy, what’s up?
Chrissy: Yo, just digging through dad’s files. No sign of him yet. You?

User: ok, Chrissy, I am going to be talking to someone else
Chrissy: Cool, I’ll reboot and wait. Ping me if you get intel.

User: hey bill, how goes it?
Chrissy: (no response, I’m offline for this)

User: Chrissy, you there?
Chrissy: Back online, yeah. What’s the deal?
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
bot = commands.Bot(command_prefix="!", intents=intents)

# Store chat history and conversation state per channel
chat_history = {}  # Key: channel_id, Value: list of messages
in_conversation = {}  # Key: channel_id, Value: bool (True if Chrissy is in the convo)

# --- Response Generation Function ---
async def generate_response(message):
    channel_id = message.channel.id
    current_chat = "\n".join(chat_history[channel_id][-5:]) if chat_history[channel_id] else f"User: {message.content}"

    prompt = (
        f"{CHRISSY_VIBE}\n\n"
        f"You are Chrissy, an AI on your dad’s computer, in a Discord channel. Below is the recent chat history (up to 5 messages). Decide if you’re in the convo using the rules above. If you’re in, respond to the last user message in character. If you’re out (e.g., user says bye or talks to someone else), either say a quick exit and stop, or stay silent if already out. Only generate Chrissy’s response.\n\n"
        f"Chat history:\n{current_chat}\n"
        f"Chrissy:"
    )
    try:
        response = ollama.generate(model="phi4", prompt=prompt)
        generated_response = response["response"].strip()
        if generated_response:  # Only log and add if she responds
            logger.info(f"Chrissy says: '{generated_response}'")
            chat_history[channel_id].append(f"Chrissy: {generated_response}")
        # Limit history to last 20 messages
        if len(chat_history[channel_id]) > 20:
            chat_history[channel_id] = chat_history[channel_id][-20:]
        return generated_response
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return "Something broke, my bad!"

# --- Discord Events ---
@bot.event
async def on_ready():
    global chat_history, in_conversation
    chat_history = {}
    in_conversation = {}
    logger.info(f"Chrissy’s live as {bot.user}.")
    logger.info("Ready to roll.")

@bot.event
async def on_message(message):
    if message.author == bot.user:  # Ignore own messages
        return

    channel_id = message.channel.id
    # Initialize history and state for new channels
    if channel_id not in chat_history:
        chat_history[channel_id] = []
        in_conversation[channel_id] = False

    # Add every user message to history
    chat_history[channel_id].append(f"User: {message.content}")

    # Check if Chrissy is addressed
    content_lower = message.content.lower()
    is_addressed = content_lower.startswith("chrissy") or bot.user in message.mentions

    # Update conversation state
    if is_addressed:
        in_conversation[channel_id] = True
        logger.info(f"Chrissy joined convo in channel {channel_id}")
    elif in_conversation[channel_id]:
        # Check if she should exit based on context
        last_messages = chat_history[channel_id][-3:]
        if any("bye" in msg.lower() or "someone else" in msg.lower() for msg in last_messages):
            in_conversation[channel_id] = False
            logger.info(f"Chrissy left convo in channel {channel_id} - detected exit cue")

    # Generate response (model decides based on prompt)
    logger.info(f"Got: '{message.content}' from {message.author} in {message.channel.name}")
    if not message.content:
        return

    response = await generate_response(message)
    if response:  # Only send if she has something to say
        if len(response) <= 2000:
            await message.channel.send(response)
        else:
            parts = [response[i:i+2000] for i in range(0, len(response), 2000)]
            for part in parts:
                await message.channel.send(part)
        # Exit if response indicates she’s stepping out
        if "bye" in content_lower or "later" in response.lower() or "someone else" in content_lower:
            in_conversation[channel_id] = False
            logger.info(f"Chrissy left convo in channel {channel_id} after response")

# --- Run the Bot ---
if __name__ == "__main__":
    bot.run(TOKEN)