import os
import glob
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch

# Initialize model and processor
model_path = "moonshotai/Kimi-VL-A3B-Thinking-2506"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

def extract_thinking_and_summary(text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> tuple:
    if bot in text and eot not in text:
        return "", ""
    if eot in text:
        thinking = text[text.index(bot) + len(bot):text.index(eot)].strip()
        summary = text[text.index(eot) + len(eot):].strip()
        return thinking, summary
    return "", text

# Get all image files from current directory
image_files = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']:
    image_files.extend(glob.glob(ext))
    image_files.extend(glob.glob(ext.upper()))

if not image_files:
    print("No image files found in current directory")
    exit()

print(f"Found {len(image_files)} image files to process")

# Process each image
for i, image_path in enumerate(image_files, 1):
    print(f"\nProcessing {i}/{len(image_files)}: {os.path.basename(image_path)}")
    
    # Load image
    image = Image.open(image_path)
    
    # Create message
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image}, 
            {"type": "text", "text": "Convert this table to HTML table format."}
        ]}
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=4096, temperature=0.8, do_sample=True)
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    thinking, summary = extract_thinking_and_summary(response)
    
    # Save the HTML output
    base_name = Path(image_path).stem
    output_path = f"{base_name}.html"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary if summary else response)
    
    print(f"Saved: {output_path}")

print("\nDone!")