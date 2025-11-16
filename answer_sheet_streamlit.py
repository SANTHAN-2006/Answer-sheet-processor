import streamlit as st
import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import uuid
import time
from streamlit_option_menu import option_menu
from streamlit_image_comparison import image_comparison
from datetime import datetime
import json
import requests
from typing import Optional, Tuple, List, Dict

# ============================================
# MOODLE CONFIGURATION
# ============================================
MOODLE_CONFIG = {
    "url": "https://05f244c11755.ngrok-free.app/webservice/rest/server.php",
    "token": "c53569d516cd601cb78849cd64f59eaa",
    "assignment_id": 1, 
    "user_id": 2,
    "timeout": 30
}

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Smart Answer Sheet Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CUSTOM CSS
# ============================================
def local_css():
    st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }

        [data-testid="stHeader"] button {
            display: none !important;
        }

        .stButton>button {
            font-weight: 500;
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            width: 100%;
            font-size: 1.1rem;
        }

        .success-box {
            background-color: #d4edda;
            border-color: #c3e6cb;
            color: #155724 !important;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        .error-box {
            background-color: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24 !important;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #cce5ff;
            border-color: #b8daff;
            color: #004085 !important;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        .warning-box {
            background-color: #fff3cd;
            border-color: #ffeeba;
            color: #856404 !important;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }

        .result-card {
            background-color: var(--secondary-background-color);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        .header-container {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            color: white;
        }

        .camera-container {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 15px;
            background-color: var(--secondary-background-color);
        }

        .image-container {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .history-item {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            background-color: var(--secondary-background-color);
            cursor: pointer;
            transition: all 0.3s;
            border-left: 5px solid var(--primary-color);
        }

        .history-item:hover {
            filter: brightness(95%);
            transform: translateY(-2px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .extracted-output {
            background-color: var(--secondary-background-color);
            border: 2px solid var(--primary-color);
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            font-family: 'Courier New', Courier, monospace;
        }

        .footer {
            margin-top: 50px;
            padding: 20px;
            text-align: center;
            font-size: 0.9rem;
            background-color: var(--secondary-background-color);
            border-radius: 10px;
            box-shadow: 0 -2px 4px rgba(0,0,0,0.05);
            width: 100%;
        }

        @media (max-width: 768px) {
            .footer { padding: 15px; font-size: 0.8rem; }
            .stButton>button { padding: 0.5rem 1rem; font-size: 1rem; }
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'image_path': None,
        'image_captured': False,
        'results_history': [],
        'processing_start_time': None,
        'selected_history_item_index': None,
        'webrtc_key': f"webrtc_{uuid.uuid4().hex}",
        'input_method': "Upload Image",
        'results': None,
        'extraction_complete': False,
        'moodle_submission_status': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================
# UTILITY FUNCTIONS
# ============================================
def st_success(text):
    st.markdown(f'<div class="success-box">✅ {text}</div>', unsafe_allow_html=True)

def st_error(text):
    st.markdown(f'<div class="error-box">❌ {text}</div>', unsafe_allow_html=True)

def st_info(text):
    st.markdown(f'<div class="info-box">ℹ️ {text}</div>', unsafe_allow_html=True)

def st_warning(text):
    st.markdown(f'<div class="warning-box">⚠️ {text}</div>', unsafe_allow_html=True)

# ============================================
# CRNN MODEL DEFINITION
# ============================================
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Dropout2d(0.3),
            nn.Conv2d(512, 512, kernel_size=(2, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ============================================
# MOODLE INTEGRATION
# ============================================
class MoodleAPI:
    """Handle all Moodle API interactions"""
    
    def __init__(self, config: Dict):
        self.url = config["url"]
        self.upload_url = config["url"].replace(
            "/webservice/rest/server.php",
            "/webservice/upload.php"
        )
        self.token = config["token"]
        self.assignment_id = config["assignment_id"]
        self.user_id = config["user_id"]
        self.timeout = config.get("timeout", 30)
    
    def make_request(self, params: Dict, files: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """Make a request to Moodle API with proper error handling"""
        try:
            if files:
                response = requests.post(self.url, params=params, files=files, timeout=self.timeout)
            else:
                response = requests.post(self.url, params=params, timeout=self.timeout)
            
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, dict) and "exception" in data:
                error_msg = data.get("message", "Unknown Moodle error")
                return False, {"error": error_msg, "details": data}
            
            return True, data
            
        except requests.exceptions.ConnectionError as e:
            return False, {"error": "Connection Error", "details": f"Cannot connect to Moodle. Error: {str(e)}"}
        except requests.exceptions.Timeout:
            return False, {"error": "Timeout Error", "details": f"Request timed out after {self.timeout} seconds"}
        except requests.exceptions.HTTPError as e:
            return False, {"error": "HTTP Error", "details": f"HTTP {response.status_code}: {str(e)}"}
        except json.JSONDecodeError:
            return False, {"error": "Invalid Response", "details": "Server returned invalid JSON"}
        except Exception as e:
            return False, {"error": "Unexpected Error", "details": str(e)}
    
    def upload_file(self, file_path: str) -> Tuple[bool, Optional[int], str]:
        """Upload file to Moodle draft area using /webservice/upload.php endpoint"""
        if not os.path.exists(file_path):
            return False, None, f"File not found: {file_path}"
        
        filename = os.path.basename(file_path)
        
        if filename.lower().endswith(('.pdf',)):
            mimetype = 'application/pdf'
        elif filename.lower().endswith(('.jpg', '.jpeg')):
            mimetype = 'image/jpeg'
        elif filename.lower().endswith('.png'):
            mimetype = 'image/png'
        else:
            mimetype = 'application/octet-stream'
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file_1': (filename, f, mimetype)}
                data = {'token': self.token}
                
                response = requests.post(
                    self.upload_url,
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    itemid = result[0].get('itemid')
                    if itemid:
                        return True, itemid, "File uploaded successfully"
                    else:
                        return False, None, "No item ID returned from upload"
                else:
                    error_msg = result.get('message', 'Unexpected upload response format')
                    return False, None, error_msg
                    
        except Exception as e:
            return False, None, f"File upload exception: {str(e)}"
    
    def submit_assignment(self, itemid: int, register_num: str, subject_code: str) -> Tuple[bool, str]:
        """Submit assignment with uploaded file"""
        submission_params = {
            "wstoken": self.token,
            "wsfunction": "mod_assign_save_submission",
            "moodlewsrestformat": "json",
            "assignmentid": self.assignment_id,
            "plugindata[files_filemanager]": itemid,
            "plugindata[onlinetext_editor][text]": f"""
            <h3>📋 Answer Sheet Submission</h3>
            <p><strong>Register Number:</strong> {register_num}</p>
            <p><strong>Subject Code:</strong> {subject_code}</p>
            <p><strong>Submitted:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}</p>
            <p style="color: #999;">✨ Submitted via Smart Scanner App</p>
            """,
            "plugindata[onlinetext_editor][format]": "1"
        }
        
        success, data = self.make_request(submission_params)
        
        if not success:
            return False, f"Submission failed: {data.get('error', 'Unknown error')}"
        
        if data is None or (isinstance(data, list) and len(data) == 0):
            return True, "Assignment submitted successfully!"
        
        if isinstance(data, dict) and "warnings" in data:
            warnings = data["warnings"]
            if warnings:
                return True, f"Submitted with warnings: {warnings}"
        
        return True, "Assignment submitted successfully!"
    
    def get_submission_status(self) -> Tuple[bool, Dict]:
        """Get current submission status"""
        params = {
            "wstoken": self.token,
            "wsfunction": "mod_assign_get_submissions",
            "moodlewsrestformat": "json",
            "assignmentids[0]": self.assignment_id
        }
        return self.make_request(params)

def submit_to_moodle_workflow(image_path: str, register_number: str, subject_code: str) -> Tuple[bool, str]:
    """Complete workflow for submitting to Moodle"""
    try:
        moodle = MoodleAPI(MOODLE_CONFIG)
        
        st.info("📤 Step 1/2: Uploading file to Moodle...")
        success, item_id, message = moodle.upload_file(image_path)
        
        if not success:
            return False, f"Upload failed: {message}"
        
        st.success(f"File uploaded! Item ID: {item_id}")
        
        st.info("📝 Step 2/2: Submitting assignment...")
        success, message = moodle.submit_assignment(item_id, register_number, subject_code)
        
        if not success:
            return False, message
        
        return True, message
        
    except Exception as e:
        return False, f"Unexpected error during submission: {str(e)}"

# ============================================
# ANSWER SHEET EXTRACTOR
# ============================================
class AnswerSheetExtractor:
    def __init__(self, yolo_improved_path, yolo_fallback_path, register_crnn_path, subject_crnn_path):
        self.script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
        
        for dir_name in ["cropped_register_numbers", "cropped_subject_codes", "results", "uploads", "captures"]:
            os.makedirs(os.path.join(self.script_dir, dir_name), exist_ok=True)
        
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                st.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                st.info("💻 Using CPU")
        except Exception as e:
            st.warning(f"Device detection error: {e}. Using CPU.")
            self.device = torch.device('cpu')
        
        self._load_yolo_models(yolo_improved_path, yolo_fallback_path)
        self._load_crnn_models(register_crnn_path, subject_crnn_path)
        
        self.register_char_map = {0: '', **{i: str(i-1) for i in range(1, 11)}}
        self.subject_char_map = {0: '', **{i: str(i-1) for i in range(1, 11)}, **{i: chr(i - 11 + ord('A')) for i in range(11, 37)}}
    
    def _load_yolo_models(self, improved_path, fallback_path):
        try:
            if not os.path.exists(improved_path):
                raise FileNotFoundError(f"Improved weights not found: {improved_path}")
            if not os.path.exists(fallback_path):
                raise FileNotFoundError(f"Fallback weights not found: {fallback_path}")
            
            self.yolo_improved_model = YOLO(improved_path)
            self.yolo_improved_model.to(self.device)
            self.yolo_fallback_model = YOLO(fallback_path)
            self.yolo_fallback_model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO models: {e}")
    
    def _load_crnn_models(self, register_path, subject_path):
        self.register_crnn_model = CRNN(num_classes=11)
        self.register_crnn_model.to(self.device)
        if not os.path.exists(register_path):
            raise FileNotFoundError(f"Register CRNN not found: {register_path}")
        try:
            checkpoint = torch.load(register_path, map_location=self.device)
            self.register_crnn_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            self.register_crnn_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load register CRNN: {e}")
        
        self.subject_crnn_model = CRNN(num_classes=37)
        self.subject_crnn_model.to(self.device)
        if not os.path.exists(subject_path):
            raise FileNotFoundError(f"Subject CRNN not found: {subject_path}")
        try:
            checkpoint = torch.load(subject_path, map_location=self.device)
            self.subject_crnn_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            self.subject_crnn_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load subject CRNN: {e}")
        
        self.register_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.subject_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def detect_regions(self, image_path, model, model_name):
        image = cv2.imread(image_path)
        if image is None:
            st.error(f"Could not load image: {image_path}")
            return [], [], None
        
        try:
            results = model(image)
        except Exception as e:
            st.error(f"Detection error with {model_name}: {e}")
            return [], [], None
        
        detections = results[0].boxes
        classes = results[0].names
        register_regions = []
        subject_regions = []
        overlay = image.copy()
        
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = classes[class_id]
            
            h, w = image.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            color = (0, 255, 0) if label == "RegisterNumber" else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            text_y = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(overlay, f"{label} {confidence:.2f}", (x1, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            padding = 10
            padded_x1 = max(0, x1 - padding)
            padded_y1 = max(0, y1 - padding)
            padded_x2 = min(w, x2 + padding)
            padded_y2 = min(h, y2 + padding)
            
            cropped_region = image[padded_y1:padded_y2, padded_x1:padded_x2]
            save_dir = os.path.join(self.script_dir, 
                                   "cropped_register_numbers" if label == "RegisterNumber" else "cropped_subject_codes")
            save_path = os.path.join(save_dir, f"{label.lower()}_{model_name}_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(save_path, cropped_region)
            
            if label == "RegisterNumber" and confidence > 0.2:
                register_regions.append((save_path, confidence))
            elif label == "SubjectCode" and confidence > 0.2:
                subject_regions.append((save_path, confidence))
        
        overlay_path = os.path.join(self.script_dir, "results", f"overlay_{model_name}_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(overlay_path, overlay)
        return register_regions, subject_regions, overlay_path
    
    def select_best_detections(self, improved_results, fallback_results):
        improved_registers, improved_subjects, improved_overlay = improved_results
        fallback_registers, fallback_subjects, fallback_overlay = fallback_results
        
        best_register = None
        best_subject = None
        best_overlay = improved_overlay
        
        if improved_registers:
            best_register = max(improved_registers, key=lambda x: x[1])
        if fallback_registers and (not best_register or best_register[1] < max(fallback_registers, key=lambda x: x[1])[1]):
            best_register = max(fallback_registers, key=lambda x: x[1])
            best_overlay = fallback_overlay
        
        if improved_subjects:
            best_subject = max(improved_subjects, key=lambda x: x[1])
        if fallback_subjects and (not best_subject or best_subject[1] < max(fallback_subjects, key=lambda x: x[1])[1]):
            best_subject = max(fallback_subjects, key=lambda x: x[1])
            best_overlay = fallback_overlay
        
        return best_register, best_subject, best_overlay
    
    def extract_text(self, image_path, model, img_transform, char_map):
        try:
            if not os.path.exists(image_path):
                return "FILE_MISSING"
            
            image = Image.open(image_path).convert('L')
            image_tensor = img_transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = model(image_tensor).squeeze(1)
                output = output.softmax(1).argmax(1)
                seq = output.cpu().numpy()
                
                prev = 0
                result = []
                for s in seq:
                    if s != 0 and s != prev:
                        result.append(char_map.get(s, '?'))
                    prev = s
            
            return ''.join(result)
        except Exception as e:
            st.error(f"Text extraction failed: {e}")
            return "ERROR"
    
    def extract_register_number(self, image_path):
        return self.extract_text(image_path, self.register_crnn_model, 
                                self.register_transform, self.register_char_map)
    
    def extract_subject_code(self, image_path):
        return self.extract_text(image_path, self.subject_crnn_model, 
                                self.subject_transform, self.subject_char_map)
    
    def process_answer_sheet(self, image_path):
        start_time = time.time()
        
        with st.spinner("🔍 Detecting regions with improved model..."):
            improved_results = self.detect_regions(image_path, self.yolo_improved_model, "improved")
        
        improved_registers, improved_subjects, _ = improved_results
        if not (improved_registers and improved_subjects):
            with st.spinner("🔄 Using fallback model..."):
                fallback_results = self.detect_regions(image_path, self.yolo_fallback_model, "fallback")
        else:
            fallback_results = ([], [], None)
        
        best_register, best_subject, best_overlay = self.select_best_detections(improved_results, fallback_results)
        
        results = []
        best_register_path = best_register[0] if best_register else None
        best_subject_path = best_subject[0] if best_subject else None
        
        if best_register:
            with st.spinner("📝 Extracting Register Number..."):
                register_number = self.extract_register_number(best_register_path)
            results.append(("Register Number", register_number))
            st.success(f"Register Number: {register_number} (Confidence: {best_register[1]:.2f})")
        else:
            st.warning("⚠️ No Register Number detected")
        
        if best_subject:
            with st.spinner("📝 Extracting Subject Code..."):
                subject_code = self.extract_subject_code(best_subject_path)
            results.append(("Subject Code", subject_code))
            st.success(f"Subject Code: {subject_code} (Confidence: {best_subject[1]:.2f})")
        else:
            st.warning("⚠️ No Subject Code detected")
        
        processing_time = time.time() - start_time
        
        if results or best_overlay:
            history_item = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "original_image_path": image_path,
                "overlay_image_path": best_overlay,
                "register_cropped_path": best_register_path,
                "subject_cropped_path": best_subject_path,
                "results": results,
                "processing_time": processing_time
            }
            st.session_state.results_history.insert(0, history_item)
        
        return results, best_register_path, best_subject_path, best_overlay, processing_time

# ============================================
# MODEL LOADING
# ============================================
@st.cache_resource
def load_extractor():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
        
        model_files = {
            'yolo_improved': 'improved_weights.pt',
            'yolo_fallback': 'weights.pt',
            'register_crnn': 'best_crnn_model.pth',
            'subject_crnn': 'best_subject_code_model.pth'
        }
        
        missing_models = []
        for name, filename in model_files.items():
            path = os.path.join(script_dir, filename)
            if not os.path.exists(path):
                missing_models.append(filename)
        
        if missing_models:
            st.error(f"❌ Missing model files: {', '.join(missing_models)}")
            st.info("Please ensure all model weight files are in the same directory as the script.")
            return None
        
        extractor = AnswerSheetExtractor(
            os.path.join(script_dir, model_files['yolo_improved']),
            os.path.join(script_dir, model_files['yolo_fallback']),
            os.path.join(script_dir, model_files['register_crnn']),
            os.path.join(script_dir, model_files['subject_crnn'])
        )
        return extractor
    except Exception as e:
        st.error(f"Failed to initialize extractor: {e}")
        return None

# ============================================
# WEBRTC CONFIGURATION
# ============================================
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]}
    ]
})

class VideoProcessor:
    def __init__(self):
        self.frame = None
        self.last_frame_time = time.time()
        self.fps = 0
        self.frame_count = 0
        self.last_processed = 0
        self.process_interval = 0.05

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        current_time = time.time()
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        self.last_processed = current_time
        self.frame_count += 1
        
        if current_time - self.last_frame_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_frame_time)
            self.last_frame_time = current_time
            self.frame_count = 0
        
        cv2.putText(img, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        h, w = img.shape[:2]
        center_x, center_y = w//2, h//2
        cv2.line(img, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 2)
        cv2.line(img, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 2)
        cv2.putText(img, "Align Sheet & Capture", (center_x - 100, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============================================
# DISPLAY FUNCTIONS
# ============================================
def display_header():
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown('<div style="font-size: 60px; text-align: center;">📄</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<h1 style="color: white;">Smart Answer Sheet Scanner</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color: white;">Automatically extract and submit to Moodle LMS</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def save_results_to_file(results, filename_prefix="results"):
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        filename = f"{filename_prefix}_{uuid.uuid4().hex}.txt"
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "w") as f:
            for label, value in results:
                f.write(f"{label}: {value}\n")
        return filepath
    except Exception as e:
        st.error(f"Failed to save results: {e}")
        return None

def get_image_download_button(image_path, filename, button_text):
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as file:
                return st.download_button(
                    label=button_text,
                    data=file,
                    file_name=filename,
                    mime="image/jpeg",
                    key=f"download_{filename.replace('.', '_')}_{uuid.uuid4().hex}"
                )
        except Exception as e:
            st.error(f"Download button error: {e}")
    return None

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    display_header()
    
    with st.spinner("🔄 Loading AI models..."):
        extractor = load_extractor()
        if extractor:
            st.success("✅ Models loaded successfully!")
        else:
            st.error("❌ Failed to load models. Please check model files.")
            st.stop()
    
    selected_tab = option_menu(
        menu_title=None,
        options=["Scan", "History", "Settings", "About"],
        icons=["camera", "clock-history", "gear", "info-circle"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "var(--secondary-background-color)", "border-radius": "10px"},
            "icon": {"color": "var(--primary-color)", "font-size": "16px"},
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "padding": "10px"},
            "nav-link-selected": {"background-color": "var(--primary-color)"}
        }
    )
    
    # ============================================
    # SCAN TAB
    # ============================================
    if selected_tab == "Scan":
        st.markdown("### 📸 Scan Answer Sheet")
        
        # CRITICAL FIX: Only show input method buttons if extraction is NOT complete
        if not st.session_state.extraction_complete:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("⬆️ Upload Image", use_container_width=True, key="btn_upload"):
                    st.session_state.input_method = "Upload Image"
                    st.session_state.image_path = None
                    st.session_state.image_captured = False
                    st.session_state.results = None
                    st.session_state.extraction_complete = False
                    st.rerun()
            with col2:
                if st.button("📸 Use Camera", use_container_width=True, key="btn_camera"):
                    st.session_state.input_method = "Use Camera"
                    st.session_state.image_path = None
                    st.session_state.image_captured = False
                    st.session_state.results = None
                    st.session_state.extraction_complete = False
                    st.session_state.webrtc_key = f"webrtc_{uuid.uuid4().hex}"
                    st.rerun()
            with col3:
                if st.button("🔄 Reset", use_container_width=True, key="btn_reset"):
                    st.session_state.image_path = None
                    st.session_state.image_captured = False
                    st.session_state.results = None
                    st.session_state.extraction_complete = False
                    st.session_state.input_method = "Upload Image"
                    st.rerun()
            
            st.markdown("---")
        
        # Show "Start New Scan" button if extraction is complete
        if st.session_state.extraction_complete:
            if st.button("🆕 Start New Scan", use_container_width=True, type="secondary", key="btn_new_scan"):
                st.session_state.image_path = None
                st.session_state.image_captured = False
                st.session_state.results = None
                st.session_state.extraction_complete = False
                st.session_state.input_method = "Upload Image"
                st.rerun()
            st.markdown("---")
        
        # Upload Image method
        if st.session_state.input_method == "Upload Image" and not st.session_state.extraction_complete:
            st.markdown("#### 📁 Upload Answer Sheet")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["png", "jpg", "jpeg"],
                key="uploader",
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
                uploads_dir = os.path.join(script_dir, "uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                
                file_extension = uploaded_file.name.split('.')[-1].lower()
                temp_path = os.path.join(uploads_dir, f"upload_{uuid.uuid4().hex}.{file_extension}")
                
                try:
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.image_path = temp_path
                    st.session_state.image_captured = True
                    st.session_state.results = None
                    st.session_state.extraction_complete = False
                    st.image(temp_path, caption="Uploaded Image", width=None)
                except Exception as e:
                    st.error(f"Error saving file: {e}")
        
        # Camera method
        elif st.session_state.input_method == "Use Camera" and not st.session_state.extraction_complete:
            if not st.session_state.image_captured:
                st.markdown("#### 📸 Camera Feed")
                st.info("Position the answer sheet and click 'Capture Image'")
                
                media_constraints = {
                    "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "frameRate": {"ideal": 30}},
                    "audio": False
                }
                
                ctx = webrtc_streamer(
                    key=st.session_state.webrtc_key,
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    media_stream_constraints=media_constraints,
                    video_processor_factory=VideoProcessor,
                    async_processing=True
                )
                
                capture_disabled = not (ctx.state.playing and ctx.video_processor)
                if st.button("📸 Capture Image", disabled=capture_disabled, use_container_width=True, key="btn_capture"):
                    if ctx.video_processor and hasattr(ctx.video_processor, 'frame') and ctx.video_processor.frame is not None:
                        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
                        captures_dir = os.path.join(script_dir, "captures")
                        os.makedirs(captures_dir, exist_ok=True)
                        temp_path = os.path.join(captures_dir, f"capture_{uuid.uuid4().hex}.jpg")
                        
                        try:
                            cv2.imwrite(temp_path, ctx.video_processor.frame)
                            if os.path.exists(temp_path):
                                st.session_state.image_path = temp_path
                                st.session_state.image_captured = True
                                st.session_state.results = None
                                st.session_state.extraction_complete = False
                                st.success("✅ Image captured!")
                                st.rerun()
                            else:
                                st.error("Failed to save captured image")
                        except Exception as e:
                            st.error(f"Capture error: {e}")
                    else:
                        st.warning("No frame available. Please wait and try again.")
            else:
                st.markdown("#### 📷 Captured Image")
                if st.session_state.image_path and os.path.exists(st.session_state.image_path):
                    st.image(st.session_state.image_path, caption="Captured Image", width=None)
                    if st.button("🔄 Recapture", use_container_width=True, key="btn_recapture"):
                        st.session_state.image_captured = False
                        st.session_state.image_path = None
                        st.session_state.results = None
                        st.session_state.extraction_complete = False
                        st.session_state.webrtc_key = f"webrtc_{uuid.uuid4().hex}"
                        st.rerun()
        
        # Show current image if extraction is complete
        if st.session_state.extraction_complete and st.session_state.image_path:
            st.markdown("#### 📷 Scanned Image")
            if os.path.exists(st.session_state.image_path):
                st.image(st.session_state.image_path, caption="Current Image", width=None)
        
        # Extract button - only show if image is ready but not yet extracted
        if (st.session_state.image_path and 
            st.session_state.image_captured and 
            not st.session_state.extraction_complete):
            
            st.markdown("---")
            if st.button("🔍 Extract Information", type="primary", use_container_width=True, key="btn_extract"):
                try:
                    progress_bar = st.progress(0, text="Starting extraction...")
                    
                    progress_bar.progress(10, text="Processing image...")
                    results, register_cropped, subject_cropped, overlay_path, processing_time = extractor.process_answer_sheet(st.session_state.image_path)
                    
                    progress_bar.progress(100, text="Complete!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    
                    # Save results to session state
                    st.session_state.results = {
                        "results_list": results,
                        "register_cropped": register_cropped,
                        "subject_cropped": subject_cropped,
                        "overlay_path": overlay_path,
                        "processing_time": processing_time
                    }
                    st.session_state.extraction_complete = True
                    # DO NOT call st.rerun() here - just let it continue
                    
                except Exception as e:
                    st.error(f"Extraction error: {e}")
                    st.info("Please try with a different image.")
        
        # Display results if extraction is complete
        if st.session_state.extraction_complete and st.session_state.results:
            results = st.session_state.results["results_list"]
            register_cropped = st.session_state.results["register_cropped"]
            subject_cropped = st.session_state.results["subject_cropped"]
            overlay_path = st.session_state.results["overlay_path"]
            processing_time = st.session_state.results["processing_time"]
            
            st.markdown("---")
            st.markdown("### 📋 Extraction Results")
            
            # Results card
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            if results:
                st.markdown('<div class="extracted-output">', unsafe_allow_html=True)
                for label, value in results:
                    st.markdown(f"**{label}:** `{value}`")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download results
                results_file = save_results_to_file(results, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                if results_file and os.path.exists(results_file):
                    with open(results_file, "rb") as file:
                        st.download_button(
                            label="📥 Download Results (.txt)",
                            data=file,
                            file_name="extracted_data.txt",
                            mime="text/plain",
                            key="download_results"
                        )
            else:
                st.warning("No information could be extracted")
            
            st.markdown(f"<p style='text-align: right;'>⏱️ Processing time: {processing_time:.2f}s</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Moodle submission
            if results:
                st.markdown("### 🎓 Submit to Moodle")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info(f"**Assignment ID:** {MOODLE_CONFIG['assignment_id']} | **User ID:** {MOODLE_CONFIG['user_id']}")
                with col2:
                    if st.button("🚀 Submit to Moodle", type="primary", use_container_width=True, key="btn_submit_moodle"):
                        register_num = next((item[1] for item in results if item[0] == "Register Number"), "N/A")
                        subject_code = next((item[1] for item in results if item[0] == "Subject Code"), "N/A")
                        
                        if st.session_state.image_path and os.path.exists(st.session_state.image_path):
                            with st.spinner("Submitting to Moodle..."):
                                success, message = submit_to_moodle_workflow(
                                    st.session_state.image_path,
                                    register_num,
                                    subject_code
                                )
                                
                                if success:
                                    st.success(f"🎉 {message}")
                                    st.balloons()
                                    st.session_state.moodle_submission_status = "success"
                                else:
                                    st.error(f"❌ {message}")
                                    st.session_state.moodle_submission_status = "failed"
                        else:
                            st.error("Image file not found")
            
            # Visual results
            st.markdown("### 🖼️ Visual Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original vs Detections**")
                if st.session_state.image_path and overlay_path:
                    if os.path.exists(st.session_state.image_path) and os.path.exists(overlay_path):
                        image_comparison(
                            img1=st.session_state.image_path,
                            img2=overlay_path,
                            label1="Original",
                            label2="Detections"
                        )
                        get_image_download_button(overlay_path, "detections.jpg", "📥 Download Detections")
            
            with col2:
                st.markdown("**Cropped Regions**")
                if register_cropped and os.path.exists(register_cropped):
                    st.image(register_cropped, caption="Register Number", width=None)
                    get_image_download_button(register_cropped, "register.jpg", "📥 Download")
                
                if subject_cropped and os.path.exists(subject_cropped):
                    st.image(subject_cropped, caption="Subject Code", width=None)
                    get_image_download_button(subject_cropped, "subject.jpg", "📥 Download")
                
                if not register_cropped and not subject_cropped:
                    st.info("No regions were cropped")
    
    # ============================================
    # HISTORY TAB
    # ============================================
    elif selected_tab == "History":
        st.markdown("### 📜 Processing History")
        
        if not st.session_state.results_history:
            st.info("No history yet. Process an answer sheet to see results here.")
        else:
            for i, item in enumerate(st.session_state.results_history):
                with st.expander(f"🕐 {item.get('timestamp', 'N/A')} - Processing Time: {item.get('processing_time', 0):.2f}s"):
                    results = item.get("results", [])
                    if results:
                        for label, value in results:
                            st.markdown(f"**{label}:** `{value}`")
                    else:
                        st.info("No results extracted")
                    
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    original_path = item.get("original_image_path")
                    overlay_path = item.get("overlay_image_path")
                    
                    with col1:
                        if original_path and overlay_path and os.path.exists(original_path) and os.path.exists(overlay_path):
                            st.markdown("**Original vs Detections**")
                            image_comparison(img1=original_path, img2=overlay_path, label1="Original", label2="Detections")
                    
                    with col2:
                        register_path = item.get("register_cropped_path")
                        subject_path = item.get("subject_cropped_path")
                        
                        if register_path and os.path.exists(register_path):
                            st.image(register_path, caption="Register Number", width=None)
                        if subject_path and os.path.exists(subject_path):
                            st.image(subject_path, caption="Subject Code", width=None)
    
    # ============================================
    # SETTINGS TAB
    # ============================================
    elif selected_tab == "Settings":
        st.markdown("### ⚙️ Moodle Configuration")
        
        st.info("Current Moodle settings are configured in the code. Update the `MOODLE_CONFIG` dictionary to change these values.")
        
        st.markdown("**Current Configuration:**")
        st.json(MOODLE_CONFIG)
        
        st.markdown("---")
        st.markdown("### 📝 How to Update Configuration")
        st.markdown("""
        1. Open the Python script in a text editor
        2. Find the `MOODLE_CONFIG` dictionary at the top of the file
        3. Update the following values:
           - `url`: Your Moodle webservice REST API URL
           - `token`: Your Moodle webservice token
           - `assignment_id`: The ID of the assignment to submit to
           - `user_id`: Your Moodle user ID
        4. Save the file and restart the Streamlit app
        """)
        
        st.markdown("---")
        st.markdown("### 🧪 Test Moodle Connection")
        if st.button("🔌 Test Connection", use_container_width=True):
            with st.spinner("Testing connection..."):
                moodle = MoodleAPI(MOODLE_CONFIG)
                success, data = moodle.get_submission_status()
                
                if success:
                    st.success("✅ Connection successful!")
                    st.json(data)
                else:
                    st.error(f"❌ Connection failed: {data.get('error', 'Unknown error')}")
                    with st.expander("Error Details"):
                        st.json(data)
    
    # ============================================
    # ABOUT TAB
    # ============================================
    elif selected_tab == "About":
        st.markdown("### ℹ️ About Smart Answer Sheet Scanner")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown('<div style="font-size: 80px; text-align: center;">🧠</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("""
            This application uses computer vision and deep learning to automatically extract 
            information from answer sheets and submit them to Moodle LMS.
            """)
        
        st.markdown("---")
        st.markdown("#### 🔧 Technologies Used")
        st.markdown("""
        - **YOLOv8**: Object detection for locating fields on answer sheets
        - **CRNN**: Text recognition for reading register numbers and subject codes
        - **Streamlit**: Web interface framework
        - **Moodle Web Services**: REST API integration for LMS submission
        - **PyTorch**: Deep learning framework
        - **OpenCV**: Image processing
        """)
        
        st.markdown("---")
        st.markdown("#### 📖 How to Use")
        st.markdown("""
        1. **Scan Tab**: Upload or capture an answer sheet image
        2. **Extract**: Click "Extract Information" to process the image
        3. **Review**: Check the extracted register number and subject code
        4. **Submit**: Click "Submit to Moodle" to upload to your LMS
        5. **History**: View past scans and results
        """)
        
        st.markdown("---")
        st.markdown("#### ⚠️ Important Notes")
        st.warning("""
        - Ensure model files are in the same directory as the script
        - Configure Moodle settings before submission
        - Image quality affects extraction accuracy
        - Always verify extracted information before submission
        """)
        
        st.markdown("---")
        st.markdown("#### 📦 Required Model Files")
        st.markdown("""
        - `improved_weights.pt` - Primary YOLO model
        - `weights.pt` - Fallback YOLO model
        - `best_crnn_model.pth` - Register number recognition
        - `best_subject_code_model.pth` - Subject code recognition
        """)
    
    # Footer
    st.markdown("---")
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("© 2025 Smart Answer Sheet Scanner | Built with Streamlit & ❤️", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
