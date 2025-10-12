FROM python:3.12-slim

WORKDIR /app
COPY README.md .
COPY requirements.txt .
COPY app.py .
COPY main.py .
COPY src/ src/
COPY data/ data/

# Установка только необходимых зависимостей для Streamlit
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]