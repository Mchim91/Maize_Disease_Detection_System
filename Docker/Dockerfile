# Start from the official Python base image.
FROM python:3.11.4
# SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# Set the current working directory to /code.
# This is where we'll put the requirements.txt file and the app directory.
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

# To fix DLL missing file problem on linux docker container
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Installing project requirements
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8080"]
