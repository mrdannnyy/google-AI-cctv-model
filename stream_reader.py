import cv2
import threading
import time

class RTSPStreamReader:
    """
    A class to read a video stream (RTSP/Webcam) asynchronously in a separate thread
    to prevent the camera's read speed from bottlenecking the main AI processing loop.
    """
    def __init__(self, stream_url, name="RTSPStream"):
        self.stream_url = stream_url
        self.name = name
        self.cap = cv2.VideoCapture(stream_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce buffer size for lower latency

        if not self.cap.isOpened():
            print(f"Error: Could not open video stream at {stream_url}")
            # Exit or raise error, depending on application needs
            raise IOError(f"Failed to open stream at {stream_url}")

        (self.grabbed, self.frame) = self.cap.read()
        self.stopped = False
        self.t = threading.Thread(target=self.update, name=self.name, args=())

    def start(self):
        """Starts the thread to continuously read frames."""
        print(f"Starting stream reader thread for: {self.name}")
        self.t.start()
        return self

    def update(self):
        """Internal method: continuously reads the next frame from the stream."""
        while not self.stopped:
            if self.cap.isOpened():
                (self.grabbed, self.frame) = self.cap.read()
            else:
                self.stopped = True # Stop if stream unexpectedly closes
            time.sleep(0.001) # Small sleep to yield to other threads

    def read(self):
        """Returns the frame most recently read from the stream."""
        return self.frame

    def stop(self):
        """Safely stops the thread and releases the video capture object."""
        self.stopped = True
        self.t.join() # Wait for the thread to finish
        self.cap.release()
        print(f"Stream reader for {self.name} stopped and released.")

# Example usage (for testing this module independently)
if __name__ == '__main__':
    # REPLACE WITH YOUR ACTUAL RTSP URL
    RTSP_URL = "rtsp://admin:6501@192.168.1.132:554/channel=1&subtype=main"
    
    # Use 0 for a local webcam, or the RTSP_URL for your camera
    # To test locally without a camera, you can point this to a short video file path.
    stream = RTSPStreamReader(RTSP_URL).start()
    
    # Give the thread a moment to start capturing
    time.sleep(2.0) 

    # Start the display loop
    while True:
        frame = stream.read()
        
        if frame is not None:
            # Display the frame
            cv2.imshow("RTSP Reader Test", frame)
            
            # Stop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Handle case where frame is temporarily None (e.g., initial read)
            time.sleep(0.01)
            
    stream.stop()
    cv2.destroyAllWindows()
