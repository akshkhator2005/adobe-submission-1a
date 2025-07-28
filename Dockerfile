# Use a Python slim-buster image for AMD64 architecture
FROM --platform=linux/amd64 python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy and install basic dependencies (PyMuPDF, numpy, langdetect)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch first and explicitly
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install sentence-transformers (will now use the CPU-only torch already installed)
# This step must be completed before download_model.py is run
RUN pip install --no-cache-dir sentence-transformers

# Copy and run script to download semantic model during build
COPY download_model.py .
RUN python download_model.py

# Copy the main processing script
COPY pdf_processor.py .

# Define the command to run the application
CMD ["python", "pdf_processor.py"]