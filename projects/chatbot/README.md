# Chatbot
Simple chatbot for fun. Thanks Patrick Loeber for nice introduction to AI chatbot. 
Please follow: https://www.youtube.com/watch?v=RpWeNzfSUHw&list=PLqnslRFeH2UrFW4AUgn-eY37qOAWQpJyg&index=1

Used the following languages and tools for AI:
- Python
- Pytorch 
- nltk as Neural Network.

Used the following tools for backend api
- Fastapi
- REST

## Setup virtual environment
Setup virtual environment by the following command
```commandline
python -m venv venv
```

For Unix
```
./venv/bin/activate
```

For Windows
```
.\venv\Scripts\activate
```

# Installation

## Install packages
```commandline
pip install -r requirements.txt
```

## Post installation
Run the following python script to download nltk data
```commandline
python post_installation.py
```

## Build and Run Dockerfile
Run the following command to build.
```commandline
docker build -t chatbot .
```

Run the following command to run Docker container by exposing port 8090.
```commandline
docker run -d --name chatbot -p 8090:80 chatbot
```

Now it's ready to access chatbot api. visit the following url to test.
```
http://localhost:8090/docs
```
