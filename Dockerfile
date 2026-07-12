FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
# Talk to Ollama running on the host (must listen on 0.0.0.0).
ENV OLLAMA_HOST=http://host.docker.internal:11434

CMD ["python", "app.py"]
