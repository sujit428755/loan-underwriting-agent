FROM python:3.11-slim
LABEL maintainer="Sujit (github.com/sujit428755)"
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src
RUN python scripts/train_model.py && python scripts/generate_samples.py
EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
