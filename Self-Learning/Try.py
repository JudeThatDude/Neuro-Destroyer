import ollama
import threading
import time
import numpy as np
import random
import socket
import struct

# Set up UDP socket for network MIDI
UDP_IP = "225.0.0.37"  # Multicast address
UDP_PORT = 21928  # Common RTP MIDI port, adjustable
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)  # TTL for multicast
print(f"UDP socket set up for {UDP_IP}:{UDP_PORT}")

# Define all 88 piano keys (A0 to C8)
piano_keys = []
notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
for octave in range(0, 8):
    for note in notes:
        if octave == 0 and note in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']:
            continue
        if octave == 7 and note not in ['A', 'A#', 'B', 'C']:
            break
        piano_keys.append(f"{note}{octave}")
piano_keys.append('C8')

# Map keys to MIDI note numbers
key_midi = {}
for i, key in enumerate(piano_keys):
    key_midi[key] = 21 + i  # MIDI note 21 = A0, up to 108 = C8

# Define C major key for harmonic consistency
c_major = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
c_major_keys = [n + str(o) for o in range(1, 7) for n in c_major]  # C1 to B6

# Memory and state
memory = []  # Past melodies for reference
successful_melodies = []  # Melodies it deems "good"
current_melody = [(['C4'], 0.5)]  # Start with middle C, half-second duration
melody_lock = threading.Lock()

# Play melody via UDP network MIDI
def play_melody(melody):
    for notes, duration in melody:
        # Send note-on messages
        for note in notes:
            if note in key_midi:
                velocity = random.randint(70, 100)  # Dynamic range
                # MIDI note-on: [status, note, velocity] (0x90 = note-on, channel 1)
                midi_msg = bytes([0x90, key_midi[note], velocity])
                sock.sendto(midi_msg, (UDP_IP, UDP_PORT))
        time.sleep(duration)
        # Send note-off messages
        for note in notes:
            if note in key_midi:
                # MIDI note-off: [status, note, velocity] (0x80 = note-off, channel 1)
                midi_msg = bytes([0x80, key_midi[note], 0])
                sock.sendto(midi_msg, (UDP_IP, UDP_PORT))

# Generate a major triad chord
def generate_triad(root):
    root_idx = piano_keys.index(root)
    third_idx = (root_idx + 4) % len(piano_keys)  # Major third
    fifth_idx = (root_idx + 7) % len(piano_keys)  # Perfect fifth
    return [piano_keys[root_idx], piano_keys[third_idx], piano_keys[fifth_idx]]

# Evaluate melody "goodness" (0-1 scale)
def evaluate_melody(melody):
    score = 0
    total_notes = 0
    
    for notes, _ in melody:
        in_key = sum(1 for note in notes if note in c_major_keys)
        score += (in_key / max(len(notes), 1)) * 0.5  # 50% weight
        total_notes += 1
        
        if len(notes) > 1:
            indices = sorted([piano_keys.index(note) for note in notes])
            intervals = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
            consonant_intervals = sum(1 for interval in intervals if interval in [3, 4, 7])
            score += (consonant_intervals / len(intervals)) * 0.5  # 50% weight
    
    return score / max(total_notes, 1) if total_notes > 0 else 0

# Self-learning melody generator
def update_melody():
    global current_melody
    with melody_lock:
        current = current_melody.copy()

    choice = random.choice(['explore', 'memory', 'triad', 'phi4'])
    possible_durations = [0.25, 0.5, 0.75, 1.0]
    
    if choice == 'explore':
        new_melody = []
        for _ in range(random.randint(3, 5)):
            num_notes = random.randint(1, 3)
            if random.random() < 0.7:
                notes = random.sample(c_major_keys, min(num_notes, len(c_major_keys)))
            else:
                notes = random.sample(piano_keys, num_notes)
            duration = random.choice(possible_durations)
            new_melody.append((notes, duration))
        reasoning = "Exploring with a bias toward C major notes and chords."
    elif choice == 'memory' and successful_melodies:
        base_melody = random.choice(successful_melodies)
        new_melody = base_melody[:]
        if random.random() < 0.5:
            idx = random.randint(0, len(new_melody)-1)
            num_notes = random.randint(1, 3)
            notes = random.sample(c_major_keys, min(num_notes, len(c_major_keys))) if random.random() < 0.7 else random.sample(piano_keys, num_notes)
            new_duration = random.choice(possible_durations)
            new_melody[idx] = (notes, new_duration)
        reasoning = f"Refining a successful melody: {', '.join([f'{",".join(n)} ({d}s)' for n, d in base_melody])}."
    elif choice == 'triad':
        root = random.choice(c_major_keys[:len(c_major_keys)-7])
        new_melody = [(generate_triad(root), random.choice(possible_durations)) for _ in range(3)]
        reasoning = f"Playing a sequence of major triads starting at {root}."
    else:
        memory_str = '\n'.join(memory[-5:]) if memory else 'No past melodies yet.'
        prompt = f"""
        Current Melody: {', '.join([f'{",".join(n)} ({d}s)' for n, d in current])}
        Piano Keys: {', '.join(piano_keys)}
        Past Melodies: {memory_str}
        Suggest a 3-5 chord melody in C major with durations (e.g., 'C4,E4,G4 0.5, G4,B4,D5 1.0').
        Output:
          New Melody: <notes duration, notes duration, ...>
          Reasoning: <text>
        """
        response = ollama.generate(model='phi4', prompt=prompt)
        response_text = response['response'].strip()
        try:
            lines = response_text.split('\n')
            new_melody_str = lines[0].split('New Melody:')[1].strip().split(', ')
            new_melody = []
            for item in new_melody_str:
                parts = item.split()
                dur = float(parts[-1])
                notes = parts[0].split(',')
                if all(note in piano_keys for note in notes) and dur > 0:
                    new_melody.append((notes, dur))
            reasoning = lines[1].split('Reasoning:')[1].strip()
        except:
            new_melody = current
            reasoning = "Phi-4 failed, sticking with the current melody."

    score = evaluate_melody(new_melody)
    if score > 0.7:
        successful_melodies.append(new_melody)
        reasoning += f" I liked this one (score: {score:.2f}) and saved it!"

    with melody_lock:
        current_melody = new_melody
    
    print(f"\nPlaying: {', '.join([f'{",".join(n)} ({d}s)' for n, d in current_melody])}")
    print(f"Reasoning: {reasoning} (Score: {score:.2f})")
    play_melody(current_melody)
    memory.append(f"Played: {', '.join([f'{",".join(n)} ({d}s)' for n, d in current_melody])} | Reasoning: {reasoning} | Score: {score:.2f}")

# Background piano player thread
def piano_player():
    while True:
        update_melody()
        time.sleep(5)

# Main self-learning loop
def run_piano_self_learning():
    print("Hi! I’m a self-learning piano AI with 88 keys and polyphony (up to 3 notes).")
    print(f"I’ll send MIDI over the network to {UDP_IP}:{UDP_PORT} for VMPK.")
    print("Steps: 1) Open VMPK, 2) Set MIDI Input to 'Network' and connect to 225.0.0.37:21928,")
    print("       3) Ensure a synth is connected. Press Ctrl+C to stop.\n")
    
    player_thread = threading.Thread(target=piano_player, daemon=True)
    player_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sock.close()
        print("\nStopped! Here’s what I learned:")
        print("Last 5 Melodies:")
        print('\n'.join(memory[-5:]))
        print("Successful Melodies:", [', '.join([f'{",".join(n)} ({d}s)' for n, d in m]) for m in successful_melodies])

if __name__ == "__main__":
    run_piano_self_learning()