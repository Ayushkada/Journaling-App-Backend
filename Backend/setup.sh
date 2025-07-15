
#!/bin/bash

echo "Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo "Downloading VADER lexicon..."
python3 -c "import nltk; nltk.download('vader_lexicon')"


echo "Environment setup complete."

source venv/bin/activate
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload


# brew install pyenv openssl

# # This sets OpenSSL explicitly for the Python build
# env PYTHON_CONFIGURE_OPTS="--with-openssl=$(brew --prefix openssl)" pyenv install 3.10.13

# pyenv virtualenv 3.10.13 jrnl-env
# pyenv activate jrnl-env

# pip install -r requirements.txt
