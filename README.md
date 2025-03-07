<h1>Facial recognition project with blink detection </h1>

<p>This project aims to implement a blink detection system using a webcam, with integrated face recognition and excessive motion filtering.</p> 
<p>The system uses a combination of computer vision models to detect faces and calculate the EAR (Eye Aspect Ratio), which is used to identify eye blinks. The blink count is adjusted to disregard camera movements and noise, ensuring a more accurate count.</p>

<h3>Features</h3>

 <p>Blink detection: The system monitors the eyes and calculates the EAR to detect blinks. If the EAR is below a certain threshold, a blink is considered to have occurred.</p>
 <p>Face Recognition: Using a pre-trained model (such as MTCNN), the system detects and recognizes faces in the camera. The model is able to identify whether the face corresponds to a known or unknown person. </p>
 <p>
    Blink count: The blink counter is incremented when several consecutive blinks are detected, but the counter is reset if the person detected changes or if the face is identified as “unknown”.</p>
    <p>Motion Filter: If the camera is moving (detected by frame difference), the blink count is paused to avoid erroneous counting caused by changes in camera position.</p>
    <p>EAR Moving Average: To avoid fluctuations in values and reduce the impact of errors, blink detection uses a moving average of the EAR, smoothing out the results.</p>
    <p>Similarity Search with FAISS: The facial embeddings of the people detected are compared with a pre-existing database using FAISS (Facebook AI Similarity Search), which allows efficient and fast searches in large collections of vectors. FAISS helps identify the closest person in the database and classify them as known or unknown.</p>

 <h2>Technologies used</h2>
 <p>OpenCV: For video capture, motion detection and drawing boxes and text in the frame.</p>
 <p> dlib: For detecting facial landmarks and calculating EAR.</p>
  <p>MTCNN: For face detection.</p>  
  <p> PyTorch: For the face recognition model and face embedding.</p>
  <p>FAISS (Facebook AI Similarity Search): For performing efficient searches of facial embeddings in the database, identifying the most similar person based on the feature vectors generated.
  </p>
  <p> NumPy: For mathematical calculations and manipulating arrays.</p>
   
   <h3>Install</h3> 
   <p3>The shape_predictor_68_face_landmarks is a pre-trained model provided by dlib that identifies facial landmarks (reference points on the face, such as eyes, nose, mouth, etc.).</p3>
                       
    !wget -O shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    !bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
    
    sudo apt update
    sudo apt install libjpeg-dev libpng-dev
    pip install faiss-cpu or pip install faiss-gpu-cu12 # CUDA 12.x, Python 3.8+
    pip install opencv-python dlib torch torchvision numpy facenet_pytorch

  <h4>Turn on your webcam and run the script</h4>
                  
    python FacialRecognition1.py  
  
    
