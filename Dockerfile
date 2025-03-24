FROM python:3.9-slim

# Set environment variable to avoid Python buffering issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


COPY requirements.txt /app/

RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY . /app

EXPOSE 5050

CMD ["python", "app.py"]
