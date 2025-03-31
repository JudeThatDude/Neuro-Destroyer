import discord
from discord.ext import commands
import asyncio
import logging
import os

# --- Logging Setup ---
logger = logging.getLogger("StopChrissy")
logger.setLevel(logging.INFO)
f_handler = logging.FileHandler('stop_bot.log', mode='w', encoding='utf-8')
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.handlers.clear()
logger.addHandler(f_handler)

# --- Read Token ---
logger.info("Reading token...")
TOKEN_FILE = "token.txt"
if not os.path.exists(TOKEN_FILE):
    logger.error(f"Token file '{TOKEN_FILE}' missing.")
    exit(1)
with open(TOKEN_FILE, "r") as f:
    TOKEN = f.readline().strip() or exit(logger.error("Token file empty."))
logger.info("Token loaded.")

# --- Discord Bot Setup ---
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# --- Event to Stop Bot ---
@bot.event
async def on_ready():
    logger.info(f"Connected as {bot.user}. Shutting down...")
    await bot.close()
    logger.info("Bot disconnected.")

# --- Run Bot and Stop ---
async def main():
    logger.info("Starting bot to stop it...")
    try:
        await asyncio.wait_for(bot.start(TOKEN), timeout=30)  # 30s timeout
    except asyncio.TimeoutError:
        logger.error("Connection timed out after 30s.")
        await bot.close()
    except Exception as e:
        logger.error(f"Failed to start/stop bot: {e}")
        await bot.close()

if __name__ == "__main__":
    logger.info("Running stop script...")
    asyncio.run(main())
    logger.info("Script complete.")