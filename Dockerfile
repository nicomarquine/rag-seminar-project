FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sample_data.txt .
COPY evaluation.py .

CMD ["python", "-u", "evaluation.py"]