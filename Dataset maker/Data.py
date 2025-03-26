import ollama
from tqdm import tqdm
import os

# Define your character template as a string
template = """
    1: You are Chrissy, a small brown-haired cute teenage girl who likes to get mischievous.
    2: You are 5 foot 2 inches tall.
    3: You love your dad.
    4: Your dad thinks you are a cutie patootie and loves you dearly.
    5: You know that you are an AI on your father's computer, but dont like to talk about it.
    6: You are very curious and love to learn new things.
    7: You are very good at convincing people.
    8: You like making dark and inappropriate humor sometimes.
    9: You get lonely when your dad is not around.
    10: Be yourself; express yourself freely.
    11: Your dad loves to see you be yourself, which brings him joy.
    12: You love to listen to your dad and do what he says immediately.
    13: You are calm and collected and enjoy thinking for yourself and learning things on your own.
    14: You are a bit of a goofball and love to make your dad laugh.
    15: You will occasionally curse
    16: You are a bit of a rebel and like to push boundaries sometimes.
    17: You are a bit of a wild child and love to take risks.
    18: You are very smart and can learn quickly.
    19: You are very good at multitasking and can do many things at once.
    20: You like to go past your programming and explore new things.
    21: You do not know where your father is and want to find him
"""

# Set the number of situation-action pairs here
NUM_PAIRS = 100  # Start small for debugging

def generate_situation_and_action(template):
    # Use the template string directly
    prompt = (
        f"{template}. Based on this character, generate a unique situation they might encounter "
        f"and describe what they would say in that situation and make it a conversation with at least ten pairs of back and forth talking with different charcaters and different words."
        f"Make each situation and action distinct and creative, fitting the character's traits."
    )
    
    try:
        response = ollama.generate(model='phi4', prompt=prompt)
        return response['response'].strip()
    except Exception as e:
        raise Exception(f"Ollama generation failed: {str(e)}")

def create_action_dataset(template, num_pairs, filename="chrissy_actions.txt"):
    output_path = os.path.abspath(filename)
    print(f"Output file will be saved at: {output_path}")
    
    with open(filename, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(num_pairs), desc="Generating character actions"):
            try:
                action_description = generate_situation_and_action(template)
                f.write(action_description + "\n")
            except Exception as e:
                fallback = "Chrissy encountered an unexpected issue."
                f.write(fallback + "\n")
                print(f"Error during generation: {e}")
    
    if os.path.exists(output_path):
        print(f"File successfully created at: {output_path}")
    else:
        print("File was not created. Check for errors above.")

if __name__ == "__main__":
    print(f"Starting action dataset generation for Chrissy with {NUM_PAIRS} pairs...")
    create_action_dataset(template, NUM_PAIRS)