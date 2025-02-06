# FastAPI Cloth Classifier

This repository hosts a FastAPI server for deploying a machine learning model trained on the **DeepFashion2** dataset. The model is designed to classify clothing items from input images.

## Features

- **FastAPI Backend**: Provides a high-performance API for serving model predictions.
- **DeepFashion2 Model**: Trained on a comprehensive dataset for accurate clothing classification.
- **Prediction Endpoint**: Accepts image input via form-data and returns classification results.

## API Endpoints

### `POST /predict`

- **Description**: Takes an image file as input and returns the model's prediction.
- **Request Format**: `multipart/form-data`
- **Response Format**: JSON with classification results.

## Running the Server

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- FastAPI
- Uvicorn
- Required dependencies (listed in `requirements.txt`)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/BrandonRafaelLovelyno/fastapi-cloth-classifier.git
cd fastapi-cloth-classifier
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

4. Access the API documentation at:
   - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Deployment

### Docker

A Docker image for this repository is available:

```bash
docker pull brandonrafaellovelyno/outfit-ai-be
```

To run the container:

```bash
docker run -p 8000:8000 brandonrafaellovelyno/outfit-ai-be
```

For deployment, consider using Docker. You can find details about this project and my experience with Docker in my LinkedIn post: [LinkedIn Post](https://www.linkedin.com/posts/brandon-rafael-lovelyno_docker-learningjourney-computervision-activity-7277248706687418368-WySt?utm_source=share&utm_medium=member_desktop)

## License

This project is open-source under the MIT License.

---

Happy coding! 🚀

