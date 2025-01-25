import nmap
import time
import os
from dotenv import set_key, load_dotenv

# Load the .env file
dotenv_path = ".env"
load_dotenv(dotenv_path)

def find_camera_ip(subnet="192.168.29.0/24"):

    """
    Scans the network for a camera device with a specific vendor.
    
    Args:
        subnet (str): The subnet to scan (default is "192.168.29.0/24").
    
    Returns:
        str: The IP address of the camera if found, else None.
    """

    while True:
        scanner = nmap.PortScanner()
        scanner.scan(hosts=subnet, arguments="-p 554 -sT")  # Scanning for RTSP port

        try:
            for host in scanner.all_hosts():
                print(host)
                print(scanner[host])
                print(scanner[host]['vendor'])

                if list(scanner[host]['vendor'].values())[0] == 'Sichuan AI-Link Technology':
                    return scanner[host]['addresses']['ipv4']
                
        except KeyboardInterrupt:
            print("User Interrupted")
            break
        except:
            print("Not Found Camera...Retrying")
            time.sleep(3)

def update_camera_url(camera_ip, dotenv_path=dotenv_path):
    """
    Updates the .env file with the camera RTSP URL.
    
    Args:
        camera_ip (str): The IP address of the camera.
        dotenv_path (str): Path to the .env file.
    """
    # Get the security code from the .env file
    security_code = os.getenv("outdoor_camera_security_code")
    if not security_code:
        raise ValueError("Security code not found in .env file!")

    # Form the camera URL
    camera_url = f"rtsp://admin:{security_code}@{camera_ip}:554/h264_stream"

    # Save the camera URL to the .env file
    set_key(dotenv_path, "outdoor_camera", camera_url)
    print(f"Camera URL saved to .env: {camera_url}")