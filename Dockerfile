FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

COPY api/ api/
COPY app/ app/
COPY models/ models/
COPY training/ training/
COPY weights/ weights/
COPY data/ data/
COPY mlruns/ mlruns/
COPY mlflow.db .
EXPOSE 10000

CMD ["bash", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.port 10000 --server.address 0.0.0.0"]