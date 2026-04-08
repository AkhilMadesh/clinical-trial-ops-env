FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
