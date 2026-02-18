# 1. Base Image: Use a lightweight Python version
FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-devel

# 2. Environment Variables: Keeps Python from generating .pyc files in the container
#    and turns off buffering for easier container logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Point Matplotlib config to a writable directory (e.g., /tmp)
ENV MPLCONFIGDIR=/tmp/matplotlib_config

# 3. Install System Dependencies (Optional but recommended for Imaging)
#    'libgl1' is often needed for opencv-like libraries, and 'git' is useful.
#    If your DICOMs are compressed (e.g., JPEG2000), you might need 'libgdcm-tools' here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory inside the container
WORKDIR /cephfs/eidf212/shared/odiamant/cycle-transformer/

# 5. Copy requirements first (for better Docker caching)
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your code (dicom_to_nifti.py) into the container
#COPY . .

# # To find your user ID, run: id -u on your target system
# RUN useradd -u 28574 -m -s /bin/bash odiamant

# # Change ownership of all cache directories to the stogian user
# RUN chown -R odiamant:odiamant /home/eidf212/eidf212/odiamant/

# # Change ownership of the application directory
# RUN chown -R odiamant:odiamant /home/eidf212/eidf212/odiamant/projects/proof_of_concept_ctpa/

# # Switch to the stogian user
# USER odiamant

# 8. Define the command to run your script
#CMD ["python", "dicom_to_nifti.py"]