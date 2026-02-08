FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch FIRST
RUN pip install --no-cache-dir torch torchvision \
--index-url https://download.pytorch.org/whl/cpu

# Now install the rest
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
