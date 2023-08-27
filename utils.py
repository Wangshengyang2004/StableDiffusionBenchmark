import GPUtil
import json
import os
import logging
import pandas as pd
import datetime

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


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
    os.makedirs("reports", exist_ok=True)
    with open(f'benchmark_report_{now}.json', 'w') as f:
        json.dump(report_data, f)

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