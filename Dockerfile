FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install fastapi uvicorn torch pillow torchvision pydantic python-multipart requests
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
