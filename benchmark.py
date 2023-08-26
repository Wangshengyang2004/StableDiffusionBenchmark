
# Importing necessary libraries (Note: You'll need to install these in your environment)
import torch
from diffusers import StableDiffusionPipeline
from optimum.onnxruntime import ORTStableDiffusionPipeline  # For ONNX Runtime support
import tomesd
import argparse
import time
import psutil
import GPUtil
from tqdm import tqdm  # Importing tqdm for the progress bar
import logging
import pandas as pd
from asdff import AdPipeline
import json
import os
import datetime

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# Function to setup logging
def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

# Function to read prompts from a CSV file
def read_prompts_from_csv(csv_path):
    logging.info(f"Reading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    prompts = df['prompt'].dropna().tolist()
    negative_prompts = df['negative_prompt'].dropna().tolist()
    return prompts, negative_prompts

# Function for the actual model benchmark
def benchmark_model(pipe, prompts, negative_prompts):
    logging.info("Starting the benchmark...")
    start_time = time.time()

    # Wrapping the loop with tqdm for a progress bar
    i = 1
    for prompt, negative_prompt in tqdm(zip(prompts, negative_prompts), total=len(prompts)):
        logging.info(f"Generating image for prompt: {prompt}, negative_prompt: {negative_prompt}")
        
        image = pipe(
            prompt = prompts[i],
            negative_prompt = negative_prompts[i],
            height = 512,
            width = 512,
        ).images[0]
        save_image(image, f"image_{i}.png")
        i += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Benchmark completed.")
    
    gpu_memory = get_gpu_memory()
    
    return elapsed_time, gpu_memory

# Function to measure GPU memory
def get_gpu_memory():
    gpus = GPUtil.getGPUs()
    return gpus[0].memoryUsed  # Assuming one GPU

# Function to save image
def save_image(image, image_name):
    
    save_folder = f"output/{now}"
    os.makedirs(save_folder, exist_ok=True)
    image_path = os.path.join(save_folder, image_name)
    image.save(image_path, "PNG")
    logging.info(f"Generation Complete. Image saved to {image_path}")

# Function to generate report
def generate_report(elapsed_time, gpu_memory):
    image_resolution = (512, 512)
    report_data = {
        "Elapsed Time": f"{elapsed_time} seconds",
        "GPU Memory Used": f"{gpu_memory} MB",
        "Score": elapsed_time / gpu_memory,
        "Image Resolution": f"{image_resolution[0]} x {image_resolution[1]}"
    }
    for key, value in report_data.items():
        print(f"{key}: {value}")
    with open('benchmark_report.json', 'w') as f:
        json.dump(report_data, f)

# Main function
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    # Initialize the appropriate pipeline
    if args.yolov8_face_restoration:
        pipe = AdPipeline.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16
        ).to(device)
        # Assuming you'll apply YOLOv8 within this pipeline
    elif args.use_onnx:
        pipe = ORTStableDiffusionPipeline.from_pretrained(
            args.model_name,
            export=True
        ).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            use_safetensors=args.use_safetensors
        ).to(device)
    
    # Apply Token Merging (ToMe) if enabled
    if args.token_merging:
        tomesd.apply_patch(pipe, ratio=args.token_merging_ratio)
    
    # Apply torch.compile if enabled
    if args.torch_compile:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    
    # Enable CPU Offload if enabled
    if args.sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    
    # Enable Sliced VAE decode if enabled
    if args.sliced_vae:
        pipe.enable_vae_slicing()
    
    # Enable Full-model offloading if enabled
    if args.model_cpu_offload:
        pipe.enable_model_cpu_offload()
    
    prompts, negative_prompts = read_prompts_from_csv(args.dataset)
    # Run the benchmark
    elapsed_time, gpu_memory = benchmark_model(pipe, prompts=prompts, negative_prompts=negative_prompts)
    
    # Compute a score and generate the report
    generate_report(elapsed_time, gpu_memory)

if __name__ == "__main__":
    setup_logging()  # Setting up logging
    parser = argparse.ArgumentParser(description="Benchmark for Stable Diffusion models.")
    
    parser.add_argument("--model_name", default="runwayml/stable-diffusion-v1-5", help="Pre-trained model name")
    parser.add_argußment("--use_safetensors", default=True, help="Use safe tensors")
    
    parser.add_argument("--token_merging", action="store_true", help="Enable Token Merging")
    parser.add_argument("--token_merging_ratio", default=0.5, help="Token Merging ratio")
    
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile optimization")
    parser.add_argument("--sequential_cpu_offload", action="store_true", help="Enable CPU Offload")
    parser.add_argument("--half_precision", action="store_true", help="Enable half-precision weights")
    
    parser.add_argument("--sliced_vae", action="store_true", help="Enable Sliced VAE decode")
    parser.add_argument("--model_cpu_offload", action="store_true", help="Enable Full-model offloading")
    
    parser.add_argument("--use_onnx", action="store_true", help="Enable ONNX Runtime support")
    
    parser.add_argument("--num_images", default=20, type=int, help="Number of images to generate for the benchmark")
    
    parser.add_argument("--dataset", default="dataset.csv", type=str, help="Path to the dataset CSV file containing prompts and negative_prompts")
    
    args = parser.parse_args()
    
    # Read prompts and negative prompts from the dataset if specified
    if args.dataset:
        prompts, negative_prompts = read_prompts_from_csv(args.dataset)
    else:
        prompts, negative_prompts = [], []
    
    main(args, prompts, negative_prompts)
