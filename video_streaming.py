import cv2
from dotenv import load_dotenv
import os

def capture_video_feed():
    """
    Captures video feed from an RTSP camera stream and displays it on the screen.

    This function connects to the camera using an RTSP URL (loaded from environment variables),
    captures the video frames, resizes them to 1280x720 resolution for smoother processing,
    and displays the frames in a window. The stream continues until the user presses 'q' to exit.

    Args:
        None: The RTSP URL is retrieved from environment variables (.env file), so no arguments are needed.
        
    Returns:
        None: This function directly displays the video feed using OpenCV and does not return any value.

    Raises:
        Exception: If the camera cannot be connected or if frames cannot be retrieved, an exception will be raised.

    Environment Variables:
        - `outdoor_camera`: RTSP URL of the camera stream. This should be defined in the .env file.
    
    Example:
        capture_video_feed()
    """

    # Loading the env variables
    load_dotenv()
    rtsp_secret = os.getenv("outdoor_camera")

    # Refer README.md for the RTSP URL format

    # RTSP URL for the camera
    rtsp_url = rtsp_secret

    # Open the video stream
    #cap = cv2.VideoCapture(rtsp_url) # Using RTSP method

    # Open the video stream
    cap = cv2.VideoCapture("C:\\Users\\qaise\\Documents\\BullyCatDetector\\sample_video\\WhatsApp Video 2025-01-14 at 10.03.07 AM.mp4") # Using local video

    # Print the resolutiuon of the camera that is actually being captured
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # print(f"Initial Resolution: {width}x{height}")

    # Check if the stream opened successfully
    if not cap.isOpened():
        print("Failed to connect to the camera.")
    else:
        print("Successfully connected to the camera.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to retrieve frame.")
                break

            # Resize the frame to 1280x720
            resized_frame = cv2.resize(frame, (1280, 720))
            
            # Calculate frame resolution
            # height, width, _ = resized_frame.shape
            # print(f"Resolution: {width}x{height}")

            # Display the frame
            cv2.imshow("Camera Feed", resized_frame)
            
            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

capture_video_feed()

"""
# Print the resolutiuon of the camera that is actually being captured

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Initial Resolution: {width}x{height}")

# Resize the frame to 1280x720

    # Resize the frame to 1280x720
    resized_frame = cv2.resize(frame, (1280, 720))

# Set the resolution (lowering to 1280*720p for smoother streaming)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 1280p width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 720p height

"""