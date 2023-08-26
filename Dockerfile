# Use NVIDIA's PyTorch image as a parent image
FROM nvcr.io/nvidia/pytorch:23.07-py3

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run benchmark.py when the container launches
CMD ["python", "benchmark.py"]
