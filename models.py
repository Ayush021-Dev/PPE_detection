from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class DetectionLog(db.Model):
    __tablename__ = 'detection_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    camera_id = db.Column(db.Integer, nullable=False)
    warning_level = db.Column(db.Integer, nullable=False)  # 0=safe, 1=warning, 2=critical
    categories = db.Column(db.String(200))  # Comma-separated list of violations
    screenshot_data = db.Column(db.Text)  # Base64 encoded image
    
    def __repr__(self):
        return f'<DetectionLog {self.id}: Level {self.warning_level} at {self.timestamp}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'camera_id': self.camera_id,
            'warning_level': self.warning_level,
            'categories': self.categories,
            'screenshot_data': self.screenshot_data
        }