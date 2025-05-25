FROM huggingface/transformers-pytorch-gpu:latest

RUN apt update && apt install -y ffmpeg libsm6 libxext6
RUN pip install --upgrade pip

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
