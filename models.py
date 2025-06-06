from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class DetectionLog(db.Model):
    __tablename__ = 'detection_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    camera_id = db.Column(db.Integer, nullable=False)
    total_people = db.Column(db.Integer, nullable=False, default=0)  # Total number of people detected
    no_helmet_count = db.Column(db.Integer, nullable=False, default=0)  # Number of people without helmet
    no_boiler_count = db.Column(db.Integer, nullable=False, default=0)  # Number of people without boiler suit
    categories = db.Column(db.String(500))  # Detailed category information for each person
    screenshot_data = db.Column(db.Text)  # Base64 encoded image
    
    def __repr__(self):
        return f'<DetectionLog {self.id}: {self.total_people} people at {self.timestamp}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'camera_id': self.camera_id,
            'total_people': self.total_people,
            'no_helmet_count': self.no_helmet_count,
            'no_boiler_count': self.no_boiler_count,
            'categories': self.categories,
            'screenshot_data': self.screenshot_data
        }