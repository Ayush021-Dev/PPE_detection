from flask import Flask, render_template, Response, jsonify, request
import cv2
import torch
import numpy as np
from datetime import datetime
import base64
import os
from models import db, DetectionLog
from config import Config
import threading
import time
from ultralytics import YOLO

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# Global variables
cameras = {}  # Dictionary to store camera instances
model = None
class_names = {0: 'Boiler', 1: 'Helmet', 2: 'NO_Helmet', 3: 'No_Boiler'}
warning_lock = threading.Lock()
MAX_CAMERAS = 8  # Maximum number of cameras supported

@app.context_processor
def inject_now():
    return {'now': datetime.now()}

def load_model():
    global model
    try:
        print("Attempting to load YOLO model...")
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TrainedWih2009Model.pt')
        print(f"Model path: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        if not os.access(model_path, os.R_OK):
            raise PermissionError(f"No read permission for model file at: {model_path}")
            
        model = YOLO(model_path)
        print("Model loaded successfully")
        print(f"Model type: {type(model)}")
        print(f"Model device: {model.device}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        model = None

def initialize_cameras():
    """Initialize all cameras at startup"""
    print("Initializing cameras...")
    
    # Initialize webcam
    try:
        cameras[1] = cv2.VideoCapture(0)
        if cameras[1].isOpened():
            print("Successfully initialized Camera 1 (webcam)")
        else:
            print("Failed to open Camera 1 (webcam)")
            cameras[1] = None
    except Exception as e:
        print(f"Error initializing Camera 1: {str(e)}")
        cameras[1] = None
    
    # Initialize other cameras as offline
    for cam_id in range(2, MAX_CAMERAS + 1):
        cameras[cam_id] = None
        print(f"Camera {cam_id} initialized as offline")

def get_camera(camera_id):
    if camera_id not in cameras:
        print(f"Camera {camera_id} not found in cameras dictionary")
        return None
    return cameras[camera_id]

def detect_objects(frame):
    if model is None:
        print("Model not loaded, skipping detection")
        return []
    
    try:
        # Run YOLO detection
        results = model(frame)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                if conf > 0.3:  # Lowered confidence threshold from 0.5 to 0.3
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': cls
                    })
        
        return detections
            
    except Exception as e:
        print(f"Detection error: {e}")
        return []

def analyze_detections(detections):
    # Initialize counters
    unsafe_people = 0
    no_helmet_count = 0
    no_boiler_count = 0
    person_categories = []
    
    # Group detections by person (assuming detections close to each other are the same person)
    person_detections = []
    current_person = []
    
    for detection in detections:
        if not current_person:
            current_person.append(detection)
        else:
            # Check if this detection is close to the last one (same person)
            last_detection = current_person[-1]
            x1, y1, x2, y2 = detection['bbox']
            last_x1, last_y1, last_x2, last_y2 = last_detection['bbox']
            
            # If boxes are close enough, consider them the same person
            if (abs(x1 - last_x1) < 50 and abs(y1 - last_y1) < 50):
                current_person.append(detection)
            else:
                person_detections.append(current_person)
                current_person = [detection]
    
    if current_person:
        person_detections.append(current_person)
    
    # Analyze each person's detections
    for person in person_detections:
        person_has_helmet = False
        person_has_boiler = False
        
        for detection in person:
            class_id = detection['class_id']
            class_name = class_names[class_id]
            
            if class_name == 'Helmet':
                person_has_helmet = True
            elif class_name == 'NO_Helmet':
                person_has_helmet = False
            elif class_name == 'Boiler':
                person_has_boiler = True
            elif class_name == 'No_Boiler':
                person_has_boiler = False
        
        # Only track unsafe people
        if not person_has_helmet or not person_has_boiler:
            unsafe_people += 1
            if not person_has_helmet:
                no_helmet_count += 1
            if not person_has_boiler:
                no_boiler_count += 1
                
            # Add person's category to the list
            person_category = []
            if not person_has_helmet:
                person_category.append('No Helmet')
            if not person_has_boiler:
                person_category.append('No Boiler')
            person_categories.append(' & '.join(person_category))
    
    return {
        'total_people': unsafe_people,  # Only count unsafe people
        'no_helmet_count': no_helmet_count,
        'no_boiler_count': no_boiler_count,
        'categories': ' | '.join(person_categories)
    }

def save_screenshot_to_db(frame, analysis, camera_id=1):
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join('static', 'uploads', 'screenshots')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"screenshot_{timestamp}.jpg"
        filepath = os.path.join(upload_dir, filename)
        
        # Save the image file
        cv2.imwrite(filepath, frame)
        
        # Convert frame to base64 for database
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            raise Exception("Failed to encode image")
            
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create new log entry
        log = DetectionLog(
            timestamp=datetime.now(),
            camera_id=camera_id,
            total_people=analysis['total_people'],
            no_helmet_count=analysis['no_helmet_count'],
            no_boiler_count=analysis['no_boiler_count'],
            categories=analysis['categories'],
            screenshot_data=img_base64
        )
        
        # Save to database within application context
        with app.app_context():
            db.session.add(log)
            db.session.commit()
            print(f"Screenshot saved - ID: {log.id}, File: {filename}")
            return True
    except Exception as e:
        print(f"Failed to save screenshot: {str(e)}")
        with app.app_context():
            db.session.rollback()
        return False

def draw_bounding_boxes(frame, detections, analysis):
    # Group detections by person
    person_detections = []
    current_person = []
    
    for detection in detections:
        if not current_person:
            current_person.append(detection)
        else:
            last_detection = current_person[-1]
            x1, y1, x2, y2 = detection['bbox']
            last_x1, last_y1, last_x2, last_y2 = last_detection['bbox']
            
            if (abs(x1 - last_x1) < 50 and abs(y1 - last_y1) < 50):
                current_person.append(detection)
            else:
                person_detections.append(current_person)
                current_person = [detection]
    
    if current_person:
        person_detections.append(current_person)
    
    # Draw boxes only for unsafe people
    for person in person_detections:
        person_has_helmet = False
        person_has_boiler = False
        person_bbox = None
        
        for detection in person:
            class_id = detection['class_id']
            class_name = class_names[class_id]
            
            if class_name == 'Helmet':
                person_has_helmet = True
            elif class_name == 'NO_Helmet':
                person_has_helmet = False
            elif class_name == 'Boiler':
                person_has_boiler = True
            elif class_name == 'No_Boiler':
                person_has_boiler = False
            
            # Store the bounding box
            if person_bbox is None:
                person_bbox = detection['bbox']
        
        # Only draw box if person is unsafe
        if not person_has_helmet or not person_has_boiler:
            x1, y1, x2, y2 = person_bbox
            color = (0, 0, 255)  # Red for unsafe
            
            # Create label based on violations
            violations = []
            if not person_has_helmet:
                violations.append("No Helmet")
            if not person_has_boiler:
                violations.append("No Boiler")
            label = " & ".join(violations)
            
            # Draw thicker bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Add filled background for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x1, y1-30), (x1 + text_size[0], y1), color, -1)
            
            # Draw label with white text
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add overall status text with background
    status_text = "SAFE" if analysis['total_people'] == 0 else "WARNING"
    status_color = (0, 255, 0) if analysis['total_people'] == 0 else (0, 0, 255)
    
    # Get text size for status
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    
    # Draw status background
    cv2.rectangle(frame, (10, 10), (20 + text_size[0], 50), status_color, -1)
    
    # Draw status text
    cv2.putText(frame, status_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    return frame

def generate_frames(camera_id=1):
    print(f"Generating frames for camera {camera_id}")
    camera = get_camera(camera_id)
    warning_active = False
    warning_start_time = 0
    
    while True:
        if camera is None or not camera.isOpened():
            print(f"Camera {camera_id} is not available or not opened")
            # Return a black frame for non-initialized cameras
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue
            
        success, frame = camera.read()
        if not success:
            print(f"Failed to read frame from camera {camera_id}")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Only run detection if the camera is active
        if camera is not None and camera.isOpened():
            # Detect objects
            detections = detect_objects(frame)
            analysis = analyze_detections(detections)
            
            # Draw bounding boxes
            frame = draw_bounding_boxes(frame, detections, analysis)
            
            # Handle warnings and screenshots
            current_time = time.time()
            
            if analysis['total_people'] > 0:  # If there are unsafe people
                if not warning_active:
                    # New warning detected
                    warning_active = True
                    warning_start_time = current_time
                    print(f"\nNew warning detected on Camera {camera_id}: {analysis['categories']}")
                    # Save screenshot immediately for new warnings
                    save_screenshot_to_db(frame.copy(), analysis, camera_id)
                elif current_time - warning_start_time >= 10:  # Save every 10 seconds while warning is active
                    print(f"\nWarning ongoing on Camera {camera_id}: {analysis['categories']}")
                    save_screenshot_to_db(frame.copy(), analysis, camera_id)
                    warning_start_time = current_time
            else:
                if warning_active:
                    # Warning cleared
                    print(f"\nWarning cleared on Camera {camera_id}")
                    warning_active = False
        
        # Convert frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', current_page=0)

@app.route('/video_feed')
def video_feed():
    camera_id = request.args.get('camera_id', 1, type=int)
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def logs():
    page = request.args.get('page', 1, type=int)
    logs = DetectionLog.query.order_by(DetectionLog.timestamp.desc()).paginate(
        page=page, per_page=10, error_out=False)
    return render_template('logs.html', logs=logs)

@app.route('/api/logs')
def api_logs():
    camera_id = request.args.get('camera_id', type=int)
    query = DetectionLog.query.order_by(DetectionLog.timestamp.desc())
    
    if camera_id:
        query = query.filter_by(camera_id=camera_id)
    
    logs = query.limit(50).all()
    return jsonify([{
        'id': log.id,
        'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'camera_id': log.camera_id,
        'total_people': log.total_people,
        'no_helmet_count': log.no_helmet_count,
        'no_boiler_count': log.no_boiler_count,
        'categories': log.categories,
        'screenshot_data': log.screenshot_data if log.screenshot_data else None
    } for log in logs])

@app.route('/api/status')
def api_status():
    # Check which cameras are active
    active_cameras = {}
    for cam_id in range(1, MAX_CAMERAS + 1):
        cam = cameras.get(cam_id)
        is_active = cam is not None and cam.isOpened()
        active_cameras[cam_id] = is_active
        print(f"Camera {cam_id} status: {'Active' if is_active else 'Inactive'}")
    return jsonify({
        'model_loaded': model is not None,
        'active_cameras': active_cameras,
        'max_cameras': MAX_CAMERAS
    })

@app.route('/api/clear_logs', methods=['POST'])
def clear_logs():
    try:
        DetectionLog.query.delete()
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test_model')
def test_model():
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded',
            'model_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TrainedWih2009Model.pt'),
            'file_exists': os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TrainedWih2009Model.pt')),
            'current_dir': os.getcwd(),
            'app_dir': os.path.dirname(os.path.abspath(__file__))
        })
    return jsonify({
        'status': 'success',
        'message': 'Model loaded successfully',
        'model_type': str(type(model)),
        'model_device': str(model.device)
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    load_model()
    initialize_cameras()  # Initialize cameras at startup
    app.run(debug=True, threaded=True)