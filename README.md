## RTSP URL Format for EZVIZ Cameras

The Real-Time Streaming Protocol (RTSP) URL for accessing your EZVIZ camera’s video feed follows this format:

### Key Components
1. **`admin`**: The default administrator username for your camera.
2. **`password`**: The administrator password, which can typically be found:
   - Printed on a label behind the camera.
   - Mentioned in the user manual.
3. **`ip_address`**: The camera's IP address on your local network.
4. **`port`**: The port used for RTSP streaming. Most cameras use port `554` by default.

### How to Obtain the RTSP Details
- **Administrator Name and Password**:  
  Check the label on the camera or refer to the user manual for these credentials.

- **IP Address and Port**:  
  - Open the EZVIZ desktop or mobile application.
  - Navigate to the camera network settings to find the IP address and confirm the RTSP port.

### Example
For a camera with the following details:
- Username: `admin`
- Password: `1234`
- IP Address: `192.168.1.100`
- Port: `554`

The RTSP URL would be:
rtsp://admin:1234@192.168.1.100:554/h264_stream

# Find Camera IP by Vendor Detection Script

## Overview
This Python script is designed to identify the IP address of a camera device within a given subnet based on its vendor details. It leverages the **nmap** library to scan for devices with the **RTSP (Real Time Streaming Protocol)** port (port 554) open and checks their vendor information. 

---

## Requirements
### 1. Python Environment
Ensure Python 3.x is installed on your system.

### 2. Dependencies
Install the following dependencies:
- **nmap**: A Python library for network mapping.
- **nmap tool**: Install the nmap tool itself (needed for `nmap.PortScanner` to work). Link (https://nmap.org/download.html)
- **Add Nmap to PATH**: Search for Edit System Environment variables and add the PATH of namp under system variables as a new env variable. The PATH is mostly (C:\Program Files (x86)\Nmap)
- **Restrict Npcap driver's access to Administrators only**: Make sure to keep this unchecked during the installation process
  
Use the following commands to install:
```bash
pip install python-nmap
```