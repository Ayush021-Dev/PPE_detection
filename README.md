# 🛡️ PPE Detection System – BPCL Internship Project

A real-time, AI-powered **PPE (Personal Protective Equipment) compliance monitoring system** developed during an internship at **Bharat Petroleum Corporation Limited (BPCL), Bina Refinery – May 2025**. This Flask-based application uses a fine-tuned YOLOv8 model trained on 3,000+ custom images to monitor RTSP camera feeds and detect personnel not wearing **hard hats** or **boiler suits** in restricted zones.

The system provides a fully functional **dashboard interface**, **live camera monitoring**, and **log management**, and is hosted on **Windows IIS** for enterprise use.

---

## 🎯 Key Features

- 🧠 **YOLOv8 Object Detection**
  - Custom-trained on over 3,000 annotated PPE images
  - Detects presence of **hard hats** and **boiler suits**
  - Color-coded bounding boxes:
    - 🔵 Blue – Fully compliant
    - 🔴 Red – PPE violation (missing helmet or suit)

- 📹 **Multi-Camera Support**
  - Accepts multiple RTSP streams from refinery surveillance systems
  - Easily extendable to local/USB cameras

- 🖥️ **Flask-Based Dashboard**
  - Live camera feed view
  - Live log updates
  - Historical log viewer with:
    - 🔍 Filtering
    - 🗑️ Deletion
    - 📷 Violation snapshots

- 🗃️ **Violation Logging System**
  - Uses **SQLite + SQLAlchemy**
  - Captures:
    - Timestamp
    - Camera ID
    - Screenshot path
    - Warning type (`No Hard Hat`, `No Boiler Suit`)
    - Bounding box data (optional)

- 🧾 **Hosted on Windows IIS**
  - Integrated and deployed as a production-grade Windows service
  - Supports refinery infrastructure requirements

---

## 🧰 Tech Stack

| Component           | Technology               |
|---------------------|--------------------------|
| Model               | YOLOv8 (Ultralytics)     |
| Backend             | Flask                    |
| Deployment          | Windows IIS              |
| Database            | SQLite (via SQLAlchemy)  |
| Video Streaming     | OpenCV + RTSP support    |
| Interface           | HTML, CSS, Jinja2        |

---

## 🛠️ Installation (Local Development)

### 1. Clone the Repository
```bash
git clone https://github.com/Ayush021-Dev/PPE_detection.git
cd PPE_detection
```

### 2. Set Up Python Environment
```bash
python -m venv venv
venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Database (if not already initialized)
```bash
python db_init.py
```

### 5. Run the Flask App (Development Mode)
```bash
python app.py
```

> In production, the app is served via **Windows IIS** using `wfastcgi.py`.

---

## 🖼️ Screenshot Capture Logic

Whenever a person is detected **without required PPE**, the system:

1. Marks the bounding box **red**
2. Captures a screenshot of the frame
3. Stores the violation in the database with:
   - Timestamp
   - Camera ID
   - Screenshot path
   - Violation type

---

## 🖥️ Dashboard Overview

- **Live Feed**  
  View all configured RTSP cameras with live overlays

- **Logs**  
  - Timestamped violation list  
  - Screenshot preview  
  - Filter by camera, date, or violation type  
  - Delete unwanted logs

---

## 🧾 Database Schema (SQLite)

| Field         | Type     | Description                        |
|---------------|----------|------------------------------------|
| `id`          | Integer  | Primary Key                        |
| `timestamp`   | DateTime | Violation time                     |
| `camera_id`   | String   | ID or name of camera               |
| `violation`   | String   | `No Hard Hat`, `No Boiler Suit`    |
| `image_path`  | String   | Path to saved screenshot           |

---

## 🧪 Testing Tips

- Test with sample RTSP streams or local video files
- Verify detection by simulating PPE/no-PPE frames
- Check logs and screenshot outputs in the dashboard

---

## 📂 Project Structure

```
PPE_detection/
│
├── app.py                     # Flask app entry
├── detectors/
│   └── yolo_detector.py       # YOLOv8 inference logic
├── templates/                 # HTML templates (Jinja2)
├── static/                    # CSS, JS, saved snapshots
├── models/                    # Trained YOLOv8 weights (.pt)
├── config.py                  # RTSP camera configuration
├── database/
│   ├── db_handler.py          # SQLAlchemy logic
│   ├── models.py              # ORM models
│   └── db_init.py             # DB setup script
├── requirements.txt
└── README.md
```

---

## 🏭 Deployment at BPCL (Bina Refinery)

This system was submitted and deployed for **BPCL Bina Refinery** during my internship in May, 2025. It monitors refinery workers for mandatory PPE compliance using the plant's existing **RTSP surveillance network**, minimizing the need for manual supervision and improving safety standards.

---

## ✅ Future Improvements

- ⏰ Real-time alert notifications (email/SMS)
- 📱 Mobile dashboard (React Native/PWA)
- 🧠 Integration with refinery HR for ID tagging
- 🔒 Admin login & multi-user roles
