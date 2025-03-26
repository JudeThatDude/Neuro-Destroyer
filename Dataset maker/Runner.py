import ollama
import os

# Path to the dataset
DATASET_PATH = "chrissy_actions.txt"

def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Run the generation script first.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def act_as_chrissy(user_input, dataset):
    # Load the dataset as context
    context = (
        f"You are Chrissy, a character defined by the following behaviors:\n\n{dataset}\n\n"
        f"Based on this, respond to the following as Chrissy would, in 1-2 sentences, "
        f"matching her personality and style from the dataset: '{user_input}'"
    )
    
    # Use LLaMA 3.1 (8B) via Ollama
    response = ollama.generate(
        model='llama3.1:8b',  # Ensure this matches your Ollama model name
        prompt=context
    )
    return response['response'].strip()

def interactive_chrissy():
    print("Loading Chrissy's dataset...")
    dataset = load_dataset(DATASET_PATH)
    print("Ready! Chat with Chrissy (type 'quit' to exit):")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        try:
            response = act_as_chrissy(user_input, dataset)
            print(f"Chrissy: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    interactive_chrissy()