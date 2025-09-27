import csv
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration ---
# You can swap this with other models, but FLAN-T5 is great for instruction-based tasks.
# Other options: "distilgpt2", "google/flan-t5-base" (larger, more capable)
MODEL_NAME = "google/flan-t5-small"
INPUT_CSV_PATH = "input.csv"
OUTPUT_TEXT_PATH = "output.txt"


def load_model():
    """
    Loads the pre-trained language model and tokenizer from Hugging Face.
    Handles errors if the model files cannot be downloaded.
    
    Returns:
        tuple: A tuple containing the loaded model and tokenizer, or (None, None) on failure.
    """
    try:
        print(f"Loading model '{MODEL_NAME}'... This may take a moment.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        print("Model loaded successfully.")
        return model, tokenizer
    except (OSError, ImportError) as e:
        print(f"Error: Could not load the model '{MODEL_NAME}'.", file=sys.stderr)
        print("Please ensure you have a working internet connection to download the model files", file=sys.stderr)
        print("and that the 'transformers' and 'torch' libraries are installed correctly.", file=sys.stderr)
        print(f"Original error: {e}", file=sys.stderr)
        return None, None

def read_csv_headers(filepath):
    """
    Reads the first line of a CSV file to extract its headers.

    Args:
        filepath (str): The path to the input CSV file.

    Returns:
        list: A list of strings, where each string is a header.
              Returns an empty list if the file is not found or is empty.
    """
    try:
        with open(filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            return [header.strip() for header in headers]
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.", file=sys.stderr)
        return []
    except StopIteration:
        print(f"Error: The CSV file '{filepath}' is empty.", file=sys.stderr)
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV: {e}", file=sys.stderr)
        return []


def generate_descriptions(headers, model, tokenizer):
    """
    Generates a short description for each header using the provided language model.

    Args:
        headers (list): A list of header strings.
        model: The loaded Hugging Face language model.
        tokenizer: The loaded Hugging Face tokenizer.

    Returns:
        dict: A dictionary mapping each header to its generated description.
    """
    descriptions = {}
    print(f"\nGenerating descriptions for {len(headers)} headers...")
    for header in headers:
        # Create a clear, instructive prompt for the model.
        prompt = f"Provide a brief, one-line description for a database column named '{header}'"
        
        # Tokenize the input prompt and generate the output.
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=25, temperature=0.7)
        
        # Decode the generated tokens into a string.
        description = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        descriptions[header] = description
        print(f"  - {header} -> {description}")
        
    return descriptions


def save_to_file(descriptions, filepath):
    """
    Saves the generated header descriptions to a text file.

    Args:
        descriptions (dict): A dictionary of headers and their descriptions.
        filepath (str): The path to the output text file.
    """
    try:
        with open(filepath, mode='w', encoding='utf-8') as outfile:
            for header, desc in descriptions.items():
                outfile.write(f"{header} -> {desc}\n")
        print(f"\nResults have been successfully saved to '{filepath}'")
    except IOError as e:
        print(f"Error: Could not write to file '{filepath}'. Reason: {e}", file=sys.stderr)


def main():
    """
    Main function to orchestrate the script's workflow.
    """
    print("--- CSV Header Description Generator ---")
    
    # 1. Load the language model and tokenizer
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        sys.exit(1) # Exit if model loading failed
        
    # 2. Read headers from the input CSV file
    headers = read_csv_headers(INPUT_CSV_PATH)
    if not headers:
        print("No headers found or file could not be read. Exiting.")
        sys.exit(1)
        
    # 3. Generate descriptions for the headers
    header_descriptions = generate_descriptions(headers, model, tokenizer)
    
    # 4. Output results to a text file
    if header_descriptions:
        save_to_file(header_descriptions, OUTPUT_TEXT_PATH)
    else:
        print("Description generation failed. Nothing to save.")

if __name__ == "__main__":
    main()
