FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git && apt-get clean

# Install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY . /app
WORKDIR /app

# Expose port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false", "--server.headless=true"]

