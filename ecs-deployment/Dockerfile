# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

COPY ./requirements.txt /app

RUN pip install --upgrade pip wheel setuptools

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Run flask_chatbot_app.py when the container launches
CMD ["python", "app.py"]
