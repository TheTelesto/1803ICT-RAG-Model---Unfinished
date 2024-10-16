# Import necessary libraries
from pdfminer.high_level import extract_text
import json

# Step 1: Extract Text from PDF

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    :param pdf_path: Path to the PDF file.
    :return: Extracted text as a string.
    """
    text = extract_text(pdf_path)
    return text

# Replace 'your_coursework.pdf' with the path to your PDF file
pdf_path = "Data\Assignment.pdf"
text = extract_text_from_pdf(pdf_path)

# Step 2: Format Text into JSON

def format_text_to_json(text, json_path):
    """
    Formats extracted text into a JSON file.
    :param text: Extracted text from the PDF.
    :param json_path: Path to save the JSON file.
    """
    # Split the text into paragraphs (this is basic, and you can enhance it further)
    paragraphs = text.split('\n\n')

    # Create a JSON-like dictionary where each paragraph is an entry
    data = []
    for paragraph in paragraphs:
        if paragraph.strip():  # Exclude empty paragraphs
            data.append({
                "text": paragraph.strip()
            })

    # Convert dictionary to JSON and save to a file
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f"Saved JSON to {json_path}")

# Define the path to save the JSON file
json_path = "coursework.json"

# Format the extracted text and save it as a JSON file
format_text_to_json(text, json_path)