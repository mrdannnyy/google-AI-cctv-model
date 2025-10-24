import cv2
import time
from ultralytics import YOLO

# Import the asynchronous reader we just created
from stream_reader import RTSPStreamReader 

# --- CONFIGURATION ---
# Replace with your actual RTSP URL
RTSP_URL = "rtsp://admin:6501@192.168.1.132:554/channel=1&subtype=main"

# Model path: Use a fast YOLOv8 model for real-time
# Once you fine-tune a model on your products, replace 'yolov8n.pt' with your custom model path.
MODEL_PATH = 'yolov8n.pt' 
# ---------------------


def run_retail_analytics():
    """
    Main function to run the real-time retail AI analytics pipeline.
    """
    print("--- Starting Retail AI CCTV Application ---")
    
    # 1. Initialize YOLOv8 Model
    # YOLO automatically uses the GPU (CUDA) if available and set up correctly.
    try:
        model = YOLO(MODEL_PATH) 
        # Optional: Load a custom-trained model if available
        # model = YOLO('./models/best_retail_products.pt')
        print(f"YOLOv8 model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # 2. Start Asynchronous Video Stream
    try:
        stream = RTSPStreamReader(RTSP_URL).start()
        time.sleep(2.0) # Give the stream a moment to buffer the first frames
    except IOError:
        print("Application stopped due to stream connection error.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during stream setup: {e}")
        return


    # --- 3. Main Processing Loop ---
    start_time = time.time()
    frame_count = 0
    
    while True:
        # Get the latest frame from the separate reader thread
        frame = stream.read()

        if frame is None:
            # If the stream has stopped or is not producing frames, break
            if stream.stopped:
                print("Stream has closed unexpectedly. Exiting loop.")
                break
            time.sleep(0.01)
            continue
        
        # --- AI INFERENCE ---
        # Run the detection model on the frame
        # 'stream=True' is often used for real-time applications
        results = model(frame, stream=True, verbose=False)  
        
        # --- ANALYTICS AND VISUALIZATION ---
        for result in results:
            # The result object contains all detections (boxes, classes, confidence)
            
            # Draw the results onto the frame (YOLOv8 handles this elegantly)
            annotated_frame = result.plot()
            
            # --- RETAIL LOGIC (Placeholder for Phase 3) ---
            # This is where we will add:
            # - Object Tracking (to get customer IDs)
            # - Behavior Analysis (dwell time, suspicious action)
            # - Alert Triggering (empty shelf, queue length)
            pass 
        
        # --- Display and Performance Metrics ---
        
        # Check if the annotated frame exists (it should, after the loop)
        if 'annotated_frame' in locals():
            cv2.imshow("Retail AI Analytics Feed", annotated_frame)
        else:
            # Fallback to display the raw frame if no detections ran
            cv2.imshow("Retail AI Analytics Feed", frame)


        frame_count += 1
        
        # Calculate and display FPS (optional, but good for monitoring performance)
        if frame_count % 30 == 0: # Update FPS every 30 frames
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            print(f"Current FPS: {fps:.2f}")
            start_time = end_time


        # Stop loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Cleanup
    stream.stop()
    cv2.destroyAllWindows()
    print("Application shut down successfully.")


if __name__ == "__main__":
    run_retail_analytics()
