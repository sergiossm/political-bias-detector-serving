FROM python:3.7.13

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade -r requirements.txt

COPY src src/

RUN python src/server.py

EXPOSE 5000

CMD ["python", "src/server.py", "serve"]