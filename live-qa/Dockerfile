FROM python:latest

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Install dependecies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the files
COPY live-qa.ini ./
COPY live-qa.py ./

EXPOSE 8000
ENTRYPOINT ["uvicorn", "live-qa:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]