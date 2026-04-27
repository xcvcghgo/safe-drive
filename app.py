import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import base64

# إعدادات الصفحة
st.set_page_config(page_title="Drowsiness Detector", layout="centered")
st.title("Drowsiness Detector")

# دالة تشغيل الصوت
def play_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""<audio autoplay="true"><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>"""
        st.markdown(md, unsafe_allow_html=True)

# دالة الحساب الأصلية الخاصة بك
def calculate_ratio(points):
    A = dist.euclidean(points[1], points[5])
    B = dist.euclidean(points[2], points[4])
    C = dist.euclidean(points[0], points[3])
    return (A + B) / (2.0 * C)

# الثوابت الأصلية
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 20
MOUTH_AR_THRESH = 0.75

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.eye_close_count = 0
        self.yawn_count = 0
        self.drowsy_flag = False
        self.yawn_flag = False
        self.counter = 0
        self.is_eye_closed = False
        self.is_yawning = False

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1) # تصحيح الاتجاهات
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        self.drowsy_flag = False
        self.yawn_flag = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                LEFT_EYE = [33, 160, 158, 133, 153, 144]
                RIGHT_EYE = [362, 385, 387, 263, 373, 380]
                MOUTH = [61, 81, 13, 311, 308, 178]

                l_eye_pts = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in LEFT_EYE])
                r_eye_pts = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in RIGHT_EYE])
                mouth_pts = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in MOUTH])

                ear = (calculate_ratio(l_eye_pts) + calculate_ratio(r_eye_pts)) / 2.0
                mar = calculate_ratio(mouth_pts)

                if ear < EYE_AR_THRESH:
                    self.counter += 1
                    if not self.is_eye_closed and self.counter >= EYE_AR_CONSEC_FRAMES:
                        self.eye_close_count += 1
                        self.is_eye_closed = True
                        self.drowsy_flag = True
                else:
                    self.counter = 0
                    self.is_eye_closed = False

                if mar > MOUTH_AR_THRESH:
                    if not self.is_yawning:
                        self.yawn_count += 1
                        self.is_yawning = True
                        self.yawn_flag = True
                else:
                    self.is_yawning = False

                if self.is_eye_closed:
                    cv2.rectangle(image, (0, 0), (w, h), (0, 0, 255), 20)
                    cv2.putText(image, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        return image

# واجهة المستخدم
col1, col2 = st.columns(2)
eye_p = col1.empty()
yawn_p = col2.empty()

ctx = webrtc_streamer(key="driver-monitor", video_transformer_factory=VideoProcessor)

if ctx.video_transformer:
    eye_p.metric("Eye Close Count", ctx.video_transformer.eye_close_count)
    yawn_p.metric("Yawn Count", ctx.video_transformer.yawn_count)
    if ctx.video_transformer.drowsy_flag:
        play_audio("warning.wav")
    if ctx.video_transformer.yawn_flag:
        play_audio("yawn.wav")
