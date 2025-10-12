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

EXPOSE 7860

CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0"]