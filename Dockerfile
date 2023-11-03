FROM python:3.11-slim

# Install required system libraries
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "main.py"]