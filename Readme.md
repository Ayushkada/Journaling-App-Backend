Steps to Run:

1. Install python/ check python:
    # For Mac:
    brew install python
    # For Windows:
    Figure it out

    python3 --version
    pip --version

2. Setup Backend Environment
    cd Backend
    # For macOS/Linux:
    chmod +x setup.sh
    ./setup.sh
    source venv/bin/activate
    # For Windows (CMD):
    ./setup.bat
    venv\Scripts\activate

5. Go to APIManager.swift set baseURL with your ip (baseURL = "http://192.168.x.x:8000")

6. Run Backend:
    python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Users:
    - Users are in users.json for now, use credentials to sign in 
    - Will be moving to database eventually (Postgres or Firebase)

Journal Storage:
    - Journal entries are stored locally for now (need apple dev to test icloud but its setup)
    - Will have backup to db if user chooses
    - UI might need few changes  

Goals: 
    - storage will be same as journals 
    - need to fix UI

Insights: 
    - Nothing done yet 

ToDo:
    - UI fixes for evertything 
    - sentiment analysis 
    - apple sign in 
    - db for users, journals, and goals 
    - forgot password
    - api to get journals and goals from either local, db or icloud and feed into ai model 
    - api to take results from ai model and deliver to front end to showcase
    - integration of AI produced goals into goals page
    - some signup bugs 