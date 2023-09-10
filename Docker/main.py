import io
import uuid
import os
import numpy as np
import uvicorn
from bson import ObjectId
from fastapi import FastAPI, UploadFile, Request
from PIL import Image
from pymongo import MongoClient
from datetime import datetime
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from utils import detection

app = FastAPI()

client = MongoClient('mongodb://localhost:27017/')
db = client['Maize_disease_Classification']
collection = db['predictions']

upload_image = 'C:\\Users\\Admin\\PycharmProjects\\Docker application\\Docker\\images'
app.mount("/images", StaticFiles(directory=upload_image), name="images")

# mongo_host = "mongo"  # This corresponds to the service name in your Docker Compose
# mongo_port = 27017
# db_name = "Maize_disease_Classification"
# collection_name = "predictions"
#
# client = pymongo.MongoClient(host=mongo_host, port=mongo_port)
# db = client[db_name]
# collection = db[collection_name]

origins = [
    'http://localhost:4200',
    'https://detection-system-frontend-9xapjuzvm-mchim91.vercel.app'
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


@app.get("/get_predictions")
def get_predictions():
    predictions = list(collection.find({}))
    return {"predictions": predictions}


@app.get("/get_detection/{prediction_id}")
def get_detection(prediction_id: str):
    try:
        # Find the prediction by _id and timestamp
        prediction = collection.find_one({"_id": prediction_id})

        if prediction is not None:
            return {"prediction": prediction}
        else:
            return {"message": "Prediction not found"}
    except Exception as e:
        return {"error": f"An error occurred while fetching the prediction: {str(e)}"}


@app.post('/detection')
async def post_detection(file: UploadFile, request: Request):
    try:
        # Generate random id
        prediction_id = str(ObjectId())

        # Read image bytes from UploadFile
        image_bytes = await file.read()

        # Process the image as you were doing
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)
        image = image[:, :, ::-1].copy()
        results = detection(image)

        # Find the index with the highest confidence
        max_confidence_index = results["confidences"].index(max(results["confidences"]))

        # Get the corresponding class with the same index
        highest_confidence_class = results["classes"][max_confidence_index]

        # Get the highest confidence score
        highest_confidence_score = max(results["confidences"])

        # Convert the highest confidence score into the float range of 0 to 1
        highest_confidence_score_normalized = round(highest_confidence_score * 100)  # Round to 2 decimal places

        # Provides a list of preventions for each model being identified
        prevention_messages = {
            'Blight': ["Blight detected. Prune infected leaves and use fungicide to control spread."],
            'Rust': ["Rust identified. Use fungicide and remove infected leaves."],
            'Gray Spot': [
                "Gray spot found. Trim leaves, ensure dryness.",
                "Handle gray spot with pruning, drainage."
            ],
            'Healthy': [
                "Plant looks healthy. Maintain current care.",
                "No issues detected. Keep up care routine."
            ]
        }
        suggestion = []

        for class_name, messages in prevention_messages.items():
            if highest_confidence_class == class_name:
                suggestion = messages
                break

        filename = f"{str(uuid.uuid4())}.jpg"

        # Create the full path for the image
        image_path = os.path.join(upload_image, filename)

        # Save the uploaded bytes as an image file
        with open(image_path, "wb") as image_file:
            image_file.write(image_bytes)

        base_url = request.base_url
        image_url = f'{base_url}images/{filename}'
        # print(f"Image URL: {image_url}")
        # Store the image URL and other relevant data in MongoDB
        prediction_data = {
            "_id": prediction_id,
            "timestamp": datetime.now(),
            "prediction": results,
            "highest_confidence_class": highest_confidence_class,
            "highest_confidence_score": highest_confidence_score_normalized,
            "Suggestions": suggestion,
            "image_url": image_url  # Store the image URL
        }
        collection.insert_one(prediction_data)

        del results['boxes']

        return {
            "_id": prediction_id,
            "Class Name": highest_confidence_class,
            "Confidence Score": highest_confidence_score_normalized,
            "Suggestions": suggestion,
            "Image URL": image_url  # Return the image URL
        }
    except Exception as e:
        return {"error": f"An error occurred while processing the image: {str(e)}"}


@app.delete('/delete_prediction/{prediction_id}')
async def delete_prediction(prediction_id: str):
    try:
        # Log the received prediction_id
        # print(f"Received DELETE request for prediction_id: {prediction_id}")

        prediction = collection.find_one_and_delete({"_id": prediction_id})
        # print(f"Retrieved Prediction: {prediction}")

        if prediction is not None:
            # Delete the associated image file
            image_url = prediction.get("image_url", "")
            if image_url:
                image_path = os.path.join(upload_image, image_url.split("/")[-1])
                # print(f"Image Path: {image_path}")  # Debugging: Log the image path
                if os.path.exists(image_path):
                    os.remove(image_path)
                    # print(f"Image file deleted: {image_path}")  # Debugging: Log image deletion

            return {"success": True, "message": "Prediction deleted successfully"}
        else:
            return {"success": False, "message": "Prediction not found"}
    except Exception as e:
        return {"error": f"An error occurred while deleting the prediction: {str(e)}"}


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)
