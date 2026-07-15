FROM python:3.11-slim

WORKDIR /app

# CPU-only torch for the HHEM verifier — the default PyPI wheel drags in ~3GB
# of CUDA libraries that are useless inside this container.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
# Talk to Ollama running on the host (must listen on 0.0.0.0).
ENV OLLAMA_HOST=http://host.docker.internal:11434
# HF cache in the data volume so rebuilds don't re-download the HHEM model.
ENV HF_HOME=/app/data/hf

CMD ["python", "app.py"]
