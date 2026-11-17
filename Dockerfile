FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY assets/ ./assets/

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "src.job_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
