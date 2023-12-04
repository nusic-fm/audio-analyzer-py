FROM python:3.11-slim

# Install required system libraries
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg
RUN apt-get install -y git

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install --upgrade Flask
RUN pip install --upgrade Werkzeug

COPY . .

EXPOSE 8080
CMD ["python", "main.py"]