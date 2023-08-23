import io
import numpy as np
import uvicorn
from fastapi import FastAPI, File
from utils import detection
from PIL import Image
from pymongo import MongoClient
from datetime import datetime

client = MongoClient('mongodb://localhost:27017/')
db = client['Maize_disease_Classification']
collection = db['predictions']

# mongo_host = "mongo"  # This corresponds to the service name in your Docker Compose
# mongo_port = 27017
# db_name = "Maize_disease_Classification"
# collection_name = "predictions"
#
# client = pymongo.MongoClient(host=mongo_host, port=mongo_port)
# db = client[db_name]
# collection = db[collection_name]


app = FastAPI()


@app.post('/detection')
def post_detection(file: bytes = File(...)):
    try:
        image = Image.open(io.BytesIO(file)).convert("RGB")
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
        highest_confidence_score_normalized = highest_confidence_score

        # Provides a list of preventions for each models being identified
        prevention_messages = {
            'Blight': ["Blight detected. Prune infected leaves and use fungicide to control spread."],
            'Rust': ["Rust identified. Use fungicide and remove infected leaves."],
            'Gray Spot': ["Gray spot found. Trim leaves, ensure dryness.", "Handle gray spot with pruning, drainage."],
            'Healthy': ["Plant looks healthy. Maintain current care.", "No issues detected. Keep up care routine."]
        }
        suggestion = []

        for class_name, messages in prevention_messages.items():
            if highest_confidence_class == class_name:
                suggestion = messages
                break

        prediction_data = {
            "timestamp": datetime.now(),
            "prediction": results,
            "highest_confidence_class": highest_confidence_class,
            "highest_confidence_score": highest_confidence_score_normalized,
            "Suggestions": suggestion
        }
        collection.insert_one(prediction_data)

        del results['boxes']

        return {
            "Class Name": highest_confidence_class,
            "Confidence Score": highest_confidence_score_normalized,
            "Suggestions": suggestion
        }
    except Exception as e:
        return {"error": f"An error occurred while processing the image: {str(e)}"}


@app.get("/")
def index():
    return {"message": "Hello World"}


@app.get("/get_predictions")
def get_predictions():
    predictions = list(collection.find({}, {"_id": 0}))
    return {"predictions": predictions}


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)
