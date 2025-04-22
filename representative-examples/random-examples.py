# Load text file containing examples
def load_examples(file_path, delimiter="**_EXAMPLE_**"):
    """
    Load examples from a text file.
    Each line in the file is considered a separate example.
    """
    with open(file_path, "r") as file:
        content = file.read()

    # Split the content using the delimiter
    parts = content.split(delimiter)
    # Remove leading and trailing newlines and whitespace from each part
    # parts = [part.strip() for part in parts if part.strip()]
    return parts


examples = load_examples("samples.md")
# Print 10 random examples
import random

for i in range(10):
    example = random.choice(examples)
    print(f"{example}")
