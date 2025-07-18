import os
import glob
from pathlib import Path
from modelscope import AutoProcessor
from vllm import LLM, SamplingParams
from PIL import Image

# Initialize model and processor
model_path = "moonshotai/Kimi-VL-A3B-Thinking-2506"
llm = LLM(
    model_path,
    trust_remote_code=True,
    max_num_seqs=8,
    max_model_len=131072,
    limit_mm_per_prompt={"image": 256}
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
sampling_params = SamplingParams(max_tokens=32768, temperature=0.8)

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
            {"type": "image", "image": ""}, 
            {"type": "text", "text": "Convert this table to HTML table format."}
        ]}
    ]
    
    # Generate
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = llm.generate([{"prompt": text, "multi_modal_data": {"image": image}}], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    thinking, summary = extract_thinking_and_summary(generated_text)
    
    # Save the HTML output
    base_name = Path(image_path).stem
    output_path = f"{base_name}.html"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary if summary else generated_text)
    
    print(f"Saved: {output_path}")

print("\nDone!")