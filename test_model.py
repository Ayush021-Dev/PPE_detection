from ultralytics import YOLO
import cv2
import numpy as np

def test_model():
    try:
        # Load the model
        print("Loading YOLO model...")
        model = YOLO('TrainedWih2009Model.pt')
        print("Model loaded successfully")
        
        # Print model information
        print(f"Model type: {type(model)}")
        print(f"Model device: {model.device}")
        print(f"Model names: {model.names}")
        
        # Open webcam
        print("\nOpening webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Press 'q' to quit")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Run detection
            results = model(frame)
            
            # Process results
            for result in results:
                boxes = result.boxes
                print(f"\nDetections in frame: {len(boxes)}")
                
                # Draw boxes and labels
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Print detection info
                    print(f"Class {cls}: {model.names[cls]}, Confidence: {conf:.2f}")
                    
                    # Draw box
                    color = (0, 255, 0) if cls in [0, 1] else (0, 0, 255)  # Green for safe, Red for unsafe
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Add label
                    label = f"{model.names[cls]}: {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show frame
            cv2.imshow('YOLO Detection Test', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    test_model() 