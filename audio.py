from pydub import AudioSegment

# Load the base audio
sound = AudioSegment.from_wav("temp_base.wav")

# Pitch it up 5 semitones for that anime squeak
new_sample_rate = int(sound.frame_rate * (2 ** (5 / 12)))
pitched_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(sound.frame_rate)

# Speed it up 1.1x for a snappy vibe
sped_up_sound = pitched_sound.speedup(playback_speed=1.1)

# Normalize to kill distortion and echo from clipping
final_sound = sped_up_sound.normalize()

# Slight fade to cut abrupt ends that echo
final_sound = final_sound.fade_in(50).fade_out(50)

# Export as temp.wav, 48kHz stereo for Discord
final_sound.export("temp.wav", format="wav", parameters=["-ar", "48000", "-ac", "2"])