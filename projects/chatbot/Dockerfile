FROM python:3.11

COPY . /code

WORKDIR /code

RUN pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r /code/requirements.txt

RUN python ./utils/post_installation.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]