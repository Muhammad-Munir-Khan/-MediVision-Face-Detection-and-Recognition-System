import cv2
import numpy as np
import os
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Database file path
DATABASE_PATH = "database.pkl"

# Initialize MTCNN face detector and InceptionResnetV1 model for embeddings
mtcnn = MTCNN(keep_all=True, min_face_size=40, thresholds=[0.6, 0.7, 0.7])  # Adjusted for better detection
model = InceptionResnetV1(pretrained='vggface2').eval()


def load_database():
    """
    Load the face embeddings database from a file.
    If the file is missing or corrupt, return an empty database.
    """
    try:
        if os.path.exists(DATABASE_PATH):
            with open(DATABASE_PATH, "rb") as file:
                return pickle.load(file)
    except (pickle.UnpicklingError, EOFError):
        print("Warning: Database file is corrupt. Initializing a new database.")
    return {}


def save_database(database):
    """
    Save the face embeddings database to a file.
    """
    with open(DATABASE_PATH, "wb") as file:
        pickle.dump(database, file)


def extract_face_embedding(face_image):
    """
    Extract face embeddings using FaceNet (InceptionResnetV1 model).
    """
    # Convert face image to RGB (if not already)
    face_image_rgb = face_image[..., ::-1]

    # Detect faces using MTCNN
    faces, probs = mtcnn.detect(face_image_rgb)

    if faces is not None:
        embeddings = []
        for i, (x1, y1, x2, y2) in enumerate(faces):
            # Ensure the coordinates are within the image bounds
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Crop the face region
            face = face_image_rgb[y1:y2, x1:x2]

            # Check if the face region is valid (non-empty)
            if face.size == 0:
                print("Warning: Empty face region detected. Skipping this face.")
                continue

            # Resize the face to (160x160) to match the model's input requirements
            face_resized = cv2.resize(face, (160, 160))

            # Normalize the face to match the model's expected input
            face_resized = np.copy(face_resized)
            face_resized = face_resized / 255.0  # Normalize to [0,1]

            # Convert the numpy array to a tensor
            face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float()

            # Extract embedding using FaceNet model
            face_embedding = model(face_tensor)
            embeddings.append(face_embedding.detach().numpy())

        return embeddings
    else:
        print("Warning: No faces detected.")
        return []


def recognize_face(face_embedding, database, threshold=0.6):
    """
    Recognizes a face by comparing the embedding to the database.
    """
    best_match = ("Unknown", float("inf"))
    for name, db_embeddings in database.items():
        for db_embedding in db_embeddings:  # Compare with all embeddings of that person
            similarity = np.linalg.norm(face_embedding - db_embedding)
            if similarity < best_match[1]:
                best_match = (name, similarity)

    if best_match[1] < threshold:
        return best_match[0]
    return "Unknown"


def register_face(name, face_embedding, database):
    """
    Register a new face in the database (allow multiple samples per person).
    """
    if name in database:
        database[name].append(face_embedding)
        print(f"[INFO] New sample added for {name}.")
    else:
        database[name] = [face_embedding]
    save_database(database)
    print(f"[INFO] Face registered for {name}!")


# Main application code
def main():
    # Load the face database
    face_database = load_database()
    print("[INFO] Database loaded.")

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        faces, _ = mtcnn.detect(frame_rgb)

        if faces is not None:
            for i, (x1, y1, x2, y2) in enumerate(faces):
                # Draw bounding box around detected faces
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Crop the face and resize it
                face = frame[int(y1):int(y2), int(x1):int(x2)]

                # Extract face embedding using FaceNet
                face_embedding = extract_face_embedding(face)

                if face_embedding:
                    # Recognize the face from the database
                    name = recognize_face(face_embedding[0], face_database)
                    if name == "Unknown":
                        cv2.putText(frame, "Press 'r' to Register", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, name, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Face Detection & Recognition", frame)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('r'):  # Register new face
            new_name = input("Enter the name of the person: ").strip()
            if new_name and face_embedding:
                register_face(new_name, face_embedding[0], face_database)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
