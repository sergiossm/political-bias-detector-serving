FROM python:3.7.13

# RUN pip install torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html pillow==6.1 fastai==1.0.61 aiohttp asyncio uvicorn starlette

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["python3.7", "server.py", "serve"]