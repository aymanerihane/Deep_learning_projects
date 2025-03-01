import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import threading
import queue
import time

class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)  # Set FPS to 30
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
        
        # Initialize frame queue
        self.q = queue.Queue(maxsize=2)
        self.stopped = False
        
    def start(self):
        # Start frame collection thread
        threading.Thread(target=self.update, daemon=True).start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                return
            
            # Clear queue to ensure latest frame
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
                    
            ret, frame = self.cap.read()
            if ret:
                # Invert the frame
                frame = cv2.flip(frame, 1)  # Horizontal flip
                # frame = 255 - frame  # Invert colors
                
                # Add frame to queue
                if not self.q.full():
                    self.q.put(frame)
            time.sleep(0.001)  # Small delay to prevent CPU overload
    
    def read(self):
        return self.q.get() if not self.q.empty() else None
    
    def stop(self):
        self.stopped = True
        self.cap.release()

class FacialExpressionDetector:
    def __init__(self, model_path):
        # Initialize MediaPipe with optimized settings
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load model
        self.model = load_model(model_path)
        
        # Emotion labels
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Initialize face detectors with optimized settings
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.3,  # Lower threshold for faster detection
            model_selection=0  # Use faster model
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            refine_landmarks=False  # Disable landmark refinement for speed
        )

    def preprocess_face(self, face):
        try:
            face = cv2.resize(face, (48, 48))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)
            return face
        except Exception as e:
            return None

    def predict_emotion(self, face):
        try:
            predictions = self.model.predict(face, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = predictions[0][emotion_idx]
            return self.emotions[emotion_idx], confidence
        except Exception as e:
            return None, None

    def process_frame(self, frame, show_face_mesh=True):
        if frame is None:
            return None, []

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape
        
        # Process frame with MediaPipe
        detection_results = self.face_detection.process(rgb_frame)
        mesh_results = self.face_mesh.process(rgb_frame)
        
        face_data = []
        
        # Process detected faces
        if detection_results.detections:
            for idx, detection in enumerate(detection_results.detections):
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * frame_width))
                y = max(0, int(bbox.ymin * frame_height))
                w = min(int(bbox.width * frame_width), frame_width - x)
                h = min(int(bbox.height * frame_height), frame_height - y)
                
                if w > 0 and h > 0:
                    face = frame[y:y+h, x:x+w]
                    processed_face = self.preprocess_face(face)
                    
                    if processed_face is not None:
                        emotion, confidence = self.predict_emotion(processed_face)
                        
                        if emotion and confidence:
                            # Draw with inverted colors
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            label = f"{emotion} ({confidence:.2f})"
                            cv2.putText(frame, label, (x, y-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            
                            face_data.append({
                                'face_id': idx,
                                'emotion': emotion,
                                'confidence': confidence,
                                'position': (x, y, w, h)
                            })
        
        if show_face_mesh and mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(255, 145, 245),
                        thickness=1,
                        circle_radius=1
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(175, 0, 134),
                        thickness=1,
                        circle_radius=1
                    )
                )
        
        return frame, face_data
    
    def change_model(self, model_path):
        self.model = load_model(model_path)

    

def main():
    st.title("Multi-Face Emotion Recognition")
    
    model_options = {
        'VGG16': './vgg16_fer13_model.keras',
        'VGG19': './vgg19_model.h5',
        'EfficientNet b7': './efficientnet_model.h5'
    }
    
    model_selection = st.selectbox("Select Model", options=list(model_options.keys()))
    model_path = model_options[model_selection]
    
    show_face_mesh = st.checkbox("Show Face Mesh", value=False)
    
    try:
        detector = FacialExpressionDetector(model_path)
        
        # Initialize threaded camera stream
        stream = CameraStream().start()
        
        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        col1, col2 = st.columns(2)
        start_button = col1.button("Start")
        stop_button = col2.button("Stop")
        
        run = start_button
        
        while run and not stop_button:
            frame = stream.read()
            
            if frame is not None:
                processed_frame, face_data = detector.process_frame(frame, show_face_mesh)
                
                if processed_frame is not None:
                    frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                    
                    if face_data:
                        metrics = ""
                        for face in face_data:
                            metrics += f"Face {face['face_id']}: {face['emotion']} ({face['confidence']:.2f})\n"
                        metrics_placeholder.text(metrics)
            
            time.sleep(0.001)  # Minimal sleep to prevent CPU overload
        
        stream.stop()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
