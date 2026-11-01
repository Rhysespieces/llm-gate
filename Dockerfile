FROM python:3.12.15-slim-bookworm
RUN pip install --no-cache-dir aiohttp==3.14.5
COPY gate.py archive.py /app/
USER 1000:1000
CMD ["python", "-u", "/app/gate.py"]
