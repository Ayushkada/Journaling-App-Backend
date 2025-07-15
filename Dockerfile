FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for spaCy and NLTK
RUN apt-get update && \
    apt-get install -y gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download spaCy model and NLTK data
RUN python -m spacy download en_core_web_sm && \
    python -c "import nltk; nltk.download('vader_lexicon')"

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
