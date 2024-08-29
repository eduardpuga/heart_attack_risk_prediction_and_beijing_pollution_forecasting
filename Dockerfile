# Use a base Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /heart_attack

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port 5000 for Flask
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "heart_attack/predict_model.py"]
