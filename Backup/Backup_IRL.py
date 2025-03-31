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
    if len(chat_history[channel_id]) > 20:
        chat_history[channel_id] = chat_history[channel_id][-20:]

    content_lower = message.content.lower()
    guild = message.guild

    if "vc" in content_lower and any(word in content_lower for word in ["join", "in", "yes", "can", "could"]):
        logger.info(f"Direct VC join request detected from {message.author}")
        if guild.voice_client:
            response = await generate_response("I’m already in a VC, you deaf?", channel_id, guild)
            if response:
                await message.channel.send(response)
            return
        if message.author.voice and message.author.voice.channel:
            response = await generate_response(message.content, channel_id, guild)
            if response:
                await message.channel.send(response)
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
        return

    response = await generate_response(message.content, channel_id, guild)
    if response:
        logger.info(f"Sending response: '{response}'")
        await message.channel.send(response)
        chat_history[channel_id].append(f"Chrissy: {response}")
        save_chat_history()  # Save after responding

        if "vc" in response.lower() and guild and not guild.voice_client and message.author.voice and message.author.voice.channel:
            should_join = await should_join_vc_from_response(response, guild, channel_id)
            if should_join:
                voice_channel = message.author.voice.channel
                voice_client = await voice_channel.connect()
                voice_clients[guild.id] = voice_client
                logger.info(f"Joined VC {voice_channel.name} - Forced join due to response")
                asyncio.create_task(listen_in_vc(voice_client, guild.id))