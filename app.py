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
camera = None
model = None
class_names = {0: 'Boiler', 1: 'Helmet', 2: 'NO_Helmet', 3: 'No_Boiler'}
warning_lock = threading.Lock()

@app.context_processor
def inject_now():
    return {'now': datetime.now()}

def load_model():
    global model
    try:
        print("Attempting to load YOLO model...")
        model = YOLO('TrainedWih2009Model.pt')
        print("Model loaded successfully")
        # Print model information
        print(f"Model type: {type(model)}")
        print(f"Model device: {model.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def detect_objects(frame):
    if model is None:
        print("Model not loaded, skipping detection")
        return []
    
    try:
        print("Running detection on frame...")
        # Run YOLO detection
        results = model(frame)
        print(f"Detection results: {len(results)}")
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            print(f"Number of boxes detected: {len(boxes)}")
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                print(f"Detected class {cls} ({class_names[cls]}) with confidence {conf:.2f}")
                
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
    has_helmet = False
    has_boiler = False
    no_helmet = False
    no_boiler = False
    
    for detection in detections:
        class_id = detection['class_id']
        if class_id == 1:  # Helmet
            has_helmet = True
        elif class_id == 2:  # NO_Helmet
            no_helmet = True
        elif class_id == 0:  # Boiler
            has_boiler = True
        elif class_id == 3:  # No_Boiler
            no_boiler = True
    
    warning_level = 0
    categories = []
    
    if no_helmet:
        warning_level += 1
        categories.append('No Helmet')
    if no_boiler:
        warning_level += 1
        categories.append('No Boiler')
    
    return {
        'has_helmet': has_helmet,
        'has_boiler': has_boiler,
        'no_helmet': no_helmet,
        'no_boiler': no_boiler,
        'warning_level': warning_level,
        'categories': categories
    }

def save_screenshot_to_db(frame, analysis, camera_id=1):
    try:
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Save to database
        log = DetectionLog(
            timestamp=datetime.now(),
            camera_id=camera_id,
            warning_level=analysis['warning_level'],
            categories=', '.join(analysis['categories']),
            screenshot_data=img_base64
        )
        db.session.add(log)
        db.session.commit()
        print(f"Screenshot saved to DB with warning level {analysis['warning_level']}")
    except Exception as e:
        print(f"Error saving to DB: {e}")

def draw_bounding_boxes(frame, detections, analysis):
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        class_id = detection['class_id']
        confidence = detection['confidence']
        
        # Determine color and label based on class
        if class_id == 1:  # Helmet
            color = (0, 255, 0)  # Green
            label = f"Helmet: {confidence:.2f}"
        elif class_id == 2:  # NO_Helmet
            color = (0, 0, 255)  # Red
            label = f"No Helmet: {confidence:.2f}"
        elif class_id == 0:  # Boiler
            color = (0, 255, 0)  # Green
            label = f"Boiler Suit: {confidence:.2f}"
        elif class_id == 3:  # No_Boiler
            color = (0, 0, 255)  # Red
            label = f"No Boiler Suit: {confidence:.2f}"
        
        # Draw thicker bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Add filled background for text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x1, y1-30), (x1 + text_size[0], y1), color, -1)
        
        # Draw label with white text
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add overall status text with background
    status_text = "SAFE" if analysis['warning_level'] == 0 else "WARNING"
    status_color = (0, 255, 0) if analysis['warning_level'] == 0 else (0, 0, 255)
    
    # Get text size for status
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    
    # Draw status background
    cv2.rectangle(frame, (10, 10), (20 + text_size[0], 50), status_color, -1)
    
    # Draw status text
    cv2.putText(frame, status_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    return frame

def generate_frames():
    camera = get_camera()
    last_warning_time = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect objects
        detections = detect_objects(frame)
        analysis = analyze_detections(detections)
        
        # Draw bounding boxes
        frame = draw_bounding_boxes(frame, detections, analysis)
        
        # Check if we need to save screenshot
        current_time = time.time()
        if analysis['warning_level'] > 0 and (current_time - last_warning_time) > 5:  # 5 second cooldown
            with warning_lock:
                threading.Thread(target=save_screenshot_to_db, args=(frame.copy(), analysis)).start()
                last_warning_time = current_time
        
        # Convert frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def logs():
    page = request.args.get('page', 1, type=int)
    logs = DetectionLog.query.order_by(DetectionLog.timestamp.desc()).paginate(
        page=page, per_page=10, error_out=False)
    return render_template('logs.html', logs=logs)

@app.route('/api/logs')
def api_logs():
    logs = DetectionLog.query.order_by(DetectionLog.timestamp.desc()).limit(50).all()
    return jsonify([{
        'id': log.id,
        'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'camera_id': log.camera_id,
        'warning_level': log.warning_level,
        'categories': log.categories,
        'screenshot_data': log.screenshot_data
    } for log in logs])

@app.route('/api/status')
def api_status():
    return jsonify({
        'model_loaded': model is not None,
        'camera_active': camera is not None
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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    load_model()
    app.run(debug=True, threaded=True)