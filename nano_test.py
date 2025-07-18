import os
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from pathlib import Path

# Initialize model
model_path = "nanonets/Nanonets-OCR-s"
model = AutoModelForImageTextToText.from_pretrained(
    model_path, 
    torch_dtype="auto", 
    device_map="auto", 
    #attn_implementation="flash_attention_2"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

def ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=4096):
    prompt = """Return the table in html format. """
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

def process_images_in_folder(folder_path, output_folder="html_tables"):
    """
    Process all images in the specified folder and save HTML tables
    
    Args:
        folder_path (str): Path to folder containing images
        output_folder (str): Path to folder where HTML files will be saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # Get all image files in the folder
    image_files = []
    for file in os.listdir(folder_path):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} image files to process...")
    
    # Process each image
    for i, filename in enumerate(image_files, 1):
        try:
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {i}/{len(image_files)}: {filename}")
            
            # Convert image to HTML table
            html_result = ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=15000)
            
            # Create output filename (replace image extension with .html)
            output_filename = Path(filename).stem + ".html"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save HTML result
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_result)
            
            print(f"✓ Saved: {output_path}")
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
            continue
    
    print(f"\nProcessing complete! HTML files saved to: {output_folder}")

# Main execution
if __name__ == "__main__":
    # Specify the folder containing images
    input_folder = "table_samples"
    
    # Check if folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' does not exist!")
        print("Please make sure the 'table_samples' folder exists and contains image files.")
    else:
        # Process all images in the folder
        process_images_in_folder(input_folder)