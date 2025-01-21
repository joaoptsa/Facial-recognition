import os
import torch
from torch import nn
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
import multiprocessing
import numpy as np
import cv2
from facenet_pytorch import MTCNN
import dlib
from PIL import Image
from collections import deque
from torchvision import models
from torch.utils.data import DataLoader, random_split


#Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device configuration: {device}")

# MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True, device=device)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

#Folder configuration
data_path = "faces_dataset"
if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f"Folder '{data_path}' create.")
else:
    print(f"Folder '{data_path}' exists. Please,add images there  .")

#Transformations images
data_transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
dataset = datasets.ImageFolder(data_path, transform=data_transforms)
class_names = dataset.classes
num_classes = len(class_names)

# Model Face
class FaceRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognizer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


recognizer = FaceRecognizer(num_classes=len(class_names)).to(device)


# Generate embeddings
def generate_embeddings(dataloader):
    embeddings = []
    labels = []


    with torch.no_grad():
        for images, label in dataloader:

            images = images.to(device)
            embedding = embedder(images)
            embeddings.append(embedding)
            labels.append(label)

    return torch.cat(embeddings), torch.tensor(labels)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

# Create Dataloader
train_loader = DataLoader(
    train_data, batch_size=1, shuffle=True, num_workers=multiprocessing.cpu_count(), pin_memory=True
)
test_loader = DataLoader(
    test_data, batch_size=1, shuffle=False, num_workers=multiprocessing.cpu_count(), pin_memory=True
)

# Create DataLoader for embeddings
def create_embedding_dataloader(embeddings, labels, batch_size=1):
    dataset = torch.utils.data.TensorDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

#Generate embeddings for train and test
train_embeddings, train_labels = generate_embeddings(train_loader)
test_embeddings, test_labels = generate_embeddings(test_loader)

# Create DataLoader for train and test
train_loader = create_embedding_dataloader(train_embeddings, train_labels, batch_size=16)
test_loader = create_embedding_dataloader(test_embeddings, test_labels, batch_size=16)

# Function loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(recognizer.parameters(), lr=0.001)


# Function train  model
def train_model(epochs=10, save_path="recognizer.pth"):
    for epoch in range(epochs):
        recognizer.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings, batch_labels = batch_embeddings.to(device), batch_labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = recognizer(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calcular accuracy
            predicted = torch.argmax(outputs, dim=1)
            total_correct += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples * 100
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Save
    torch.save(recognizer.state_dict(), save_path)
    print(f"Model saved to {save_path}")



# Function test model
def test_model():
    recognizer.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for batch_embeddings, batch_labels in test_loader:
            batch_embeddings, batch_labels = batch_embeddings.to(device), batch_labels.to(device)
            outputs = recognizer(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            # accuracy
            predicted = torch.argmax(outputs, dim=1)
            total_correct += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples * 100
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")


# start train and test
train_model(epochs=10, save_path="recognizer.pth")
test_model()

############################## second part the project #######################
###############################################################################

file = "recognizer.pth"
model = FaceRecognizer(num_classes=2)  
model.load_state_dict(torch.load(file, map_location=device))
model.eval()

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

    # Detect faces
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
                face_tensor = data_transforms(face_pil).unsqueeze(0).to(device)

                embedding = embedder(face_tensor).cpu()
                ding = model(embedding).cpu()
                predicted = torch.argmax(ding, dim=1)
                label = class_names[predicted]
                if label != current_label:
                    current_label = label
                    blink_counter = 0

            except Exception as e:
                print(str(e))
                face_pil = None



            # draw box and label
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

        # show count blinks
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
