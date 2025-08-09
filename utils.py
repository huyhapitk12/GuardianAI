import base64, os, json, pickle, face_recognition
from io import BytesIO
from functools import partial
from os.path import isdir, isfile

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Pillow image encoded to base64
def p2b(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    # Encode the binary data to base64
    image = base64.b64encode(buffered.getvalue())
    # Convert to string
    base64_message = image.decode('utf-8')
    return base64_message


def get_history():
    if not os.path.isfile("Data/history.json"):
        history = [
            {
                "role": "system",
                "content": "The AI's task is to function as a virtual assistant responsible for monitoring the status of a household. When queried by the user, the AI provides information on the current condition of the home, such as the presence of people inside or potential hazards like fire. Reply in vietnamese"
            },
        ]
        
        with open("Data/history.json", "w", encoding="utf-8") as f:
            json.dump(history, f)

    else:
        with open("Data/history.json", "r", encoding="utf-8") as f:
            history = json.load(f)

    return history

def encode_known_faces(
    model: str = "cnn", encodings_location = "Processed Data/encodings.pkl") -> None:
    names = []
    encodings = []
    for i in os.listdir("Data/Image/"):
        if isdir(f"Data/Image/{i}"):
            for j in os.listdir(f"Data/Image/{i}/"):
                path = f"Data/Image/{i}/{j}"
                print(f"Processing image {path}")
                if isfile(path):
                    name = i
                    image = face_recognition.load_image_file(path)

                    face_locations = face_recognition.face_locations(image, model=model)
                    face_encodings = face_recognition.face_encodings(image,  face_locations)

                    for encoding in face_encodings:
                        names.append(name)
                        encodings.append(encoding)

        name_encodings = {"names": names, "encodings": encodings}
        with open(encodings_location, "wb") as f:
            pickle.dump(name_encodings, f)

if __name__ == "__main__":
    get_history()