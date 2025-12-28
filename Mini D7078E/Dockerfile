FROM python:3.11-slim

WORKDIR /app

COPY agent.py .

RUN pip install aiohttp

ENTRYPOINT ["python", "agent.py"]
