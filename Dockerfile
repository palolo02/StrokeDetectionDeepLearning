FROM python:3.9-slim

# Update packages in linux
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Set the working directory to /user/src/brain
WORKDIR /usr/src/brain

# Create new virtual environment
ENV VIRTUAL_ENV=python-pytorch
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
COPY requirements_pip.txt .
RUN pip install -r requirements_pip.txt

# Copy files for the application
COPY kaggle_3m ./kaggle_3m
COPY models ./models
COPY plots ./plots
COPY saved_images ./saved_images 
COPY unet ./unet 
COPY main.py .

# Run the main program
CMD ["python-pytorch/bin/python", "main.py"]
