# kreartsi-demo-deploy-ml

This repository contains code for deploying a Machine Learning model using Flask and Docker. The model is designed to predict whether an image is human-made or AI-generated.

## File and Folder Structure

- `Dockerfile`: Configuration file for Docker.
- `app.py`: The main Flask application.
- `model/`: Folder containing the trained ML model.
- `requirements.txt`: List of Python dependencies.

## Setup

Ensure you have Docker installed on your system. Clone this repository to your local machine using:

```bash
git clone [Repository URL]
```

## Running the Application

To run this application, use Docker. From the root directory of the project, execute:

```bash
docker build -t deploy-ml .
docker run -p 8080:8080 deploy-ml
```

The application will be accessible at `http://localhost:8080`.

## Using the Application

The application provides a `/predict` endpoint for predicting images using the trained ML model. Send a POST request with the image to this endpoint. The response will return the prediction and confidence level.

## Contributions

Contributions for improvements and enhancements to this application are highly welcomed. Please create issues or pull requests on this repository.
