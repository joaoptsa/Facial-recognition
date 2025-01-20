import os
import torch
from torch import nn
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
import multiprocessing
import faiss
import numpy as np
import cv2
from facenet_pytorch import MTCNN
import faiss
import dlib
from PIL import Image
from collections import deque



#Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device configuration: {device}")

#Folder configuration
data_path = "faces_dataset"
if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f"Folder '{data_path}' create.")
else:
    print(f"Folder '{data_path}' exists. Please,add images there  .")

embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

#Transformations images
data_transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load label images
dataset = datasets.ImageFolder(data_path, transform=data_transforms)
class_names = dataset.classes

loader = DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=multiprocessing.cpu_count(), pin_memory=True
)

# Function for generate embeddings
def generate_embeddings(dataloader):
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, label in dataloader:
            images = images.to(device)
            embedding = embedder(images)
            embeddings.append(embedding.cpu().numpy())
            labels.extend(label.numpy())
    return np.vstack(embeddings), np.array(labels)

# Generate embeddings and labels
train_embeddings, train_labels = generate_embeddings(loader)

# FAISS - create index
embedding_dim = train_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)  # Use distance Euclidiana (L2)

# Add embeddings índex FAISS
index.add(train_embeddings)
print(f"Index FAISS create {index.ntotal} embeddings.")

# Save faiss
#faiss.write_index(index, 'my_index.faiss')
#Load faiss
#index = faiss.read_index('my_index.faiss')

# Function  search the embedding in the índex
def search_embedding(query_embedding, k=1):
    query_embedding = query_embedding.astype(np.float32)
    distances, indices = index.search(query_embedding, k)
    return distances, indices

# Configurations
EAR_THRESHOLD = 0.3
CONSEC_FRAMES = 3
MOTION_THRESHOLD = 1000  # detection movement excess
ear_buffer = deque(maxlen=5)  # Buffer  mean móvel

# Initialized variables
blink_counter = 0
consecutive_frames = 0
prev_frame = None
current_label = None  # To track person changes

# Function  calculate the aspect ratio of the eye
def eye_aspect_ratio(eye):
    if len(eye) != 6:
        return None
    A = np.linalg.norm(eye[1] - eye[5])  # vertical distance between points 2-6
    B = np.linalg.norm(eye[2] - eye[4])  # vertical distance between points 3-5
    C = np.linalg.norm(eye[0] - eye[3])  # horizontal distance between points 1-4
    if C == 0:  #prevent against division to zero
        return None

    return (A + B) / (2.0 * C)

# Mean móvel to EAR
def calculate_smoothed_ear(ear):
    ear_buffer.append(ear)
    return sum(ear_buffer) / len(ear_buffer)

# Function for detected eyes extraction  (landmarks)
def extract_eyes(landmarks):
    LEFT_EYE_IDX = list(range(36, 42))  # Landmarks eye left
    RIGHT_EYE_IDX = list(range(42, 48))  # Landmarks eye right
    left_eye = landmarks[LEFT_EYE_IDX]
    right_eye = landmarks[RIGHT_EYE_IDX]
    return left_eye, right_eye


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mtcnn = MTCNN(keep_all=True).eval()

# open webcam
cap = cv2.VideoCapture(0)  # Capture webcam default (ID 0)

if not cap.isOpened():
    print("Error accessing the camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed capture frame.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect movement the camera
    if prev_frame is not None:
        diff = cv2.absdiff(prev_frame, gray)
        movement = np.sum(diff) / diff.size
        if movement > MOTION_THRESHOLD:
            cv2.putText(frame, "Movement detect", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Test Webcam", frame)
            prev_frame = gray
            continue
    prev_frame = gray

    boxes, _ = mtcnn.detect(frame)
    label = "unknown"
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face = frame[y1:y2, x1:x2]  # Crop da face

            if not isinstance(face, np.ndarray):
                raise TypeError(f"Wait numpy.ndarray, but receive {type(face)}")

            try:
                face_pil = Image.fromarray(cv2.resize(face, (160, 160)))
            except Exception as e:
                print(str(e))
                face_pil = None
                break

            #face_tensor = data_transforms (cv2.resize(face, (160, 160))).unsqueeze(0).to(device)
            face_tensor = data_transforms(face_pil).unsqueeze(0).to(device)
            embedding = embedder(face_tensor).cpu().detach().numpy()

            # search FAISS
            distances, indices = search_embedding(embedding)
            closest_class_index = indices[0][0]
            closest_distance = distances[0][0]
            print(closest_class_index)
            print(closest_distance)

            #Classification
            classIndex=train_labels[closest_class_index]
            label = class_names[classIndex]
            print(label)
            # Limit similarity
            if closest_distance < 0.6:
                label = label
            else:
                label = "Unknown"

            if label != current_label:
                current_label = label
                blink_counter = 0

            label = str(label)

            #draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Detection  blinks
            landmarks = predictor(gray, dlib.rectangle(x1, y1, x2, y2))
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
            left_eye, right_eye = extract_eyes(landmarks)

            # Calculate ear
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            if left_ear is not None and right_ear is not None:
                ear = calculate_smoothed_ear((left_ear + right_ear) / 2.0)

            # Count blinks
            if ear < EAR_THRESHOLD:
                consecutive_frames += 1
            else:
                if consecutive_frames >= CONSEC_FRAMES:
                    blink_counter += 1
                    print("blink detect!")
                    consecutive_frames = 0

            # Draw eyes and EAR
            for (x, y) in left_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(frame, f"EAR: {ear:.2f}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)




    cv2.imshow("Test Webcam", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close the windows  
cap.release()
cv2.destroyAllWindows()
