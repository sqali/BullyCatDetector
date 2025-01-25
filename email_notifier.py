import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
import os

# Load environment variables containing email credentials
load_dotenv()

# Server Details
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "qaiserali45@gmail.com"
SENDER_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")

def send_email_notification(image_path, confidence_score):

    """
    Sends an email notification when a bully cat is detected.

    Args:
        image_path (str): Path to the reconnaissance image to attach (currently not handled in the body; extend if needed).
        confidence_score (float): Confidence score of the detection.
    """

    subject = f"🐾 Alert! Enemy Fur-strator Detected in the Perimeter 🐾"

    body = f"""Commander of the Furry Forces, Report In!,
    
We have a situation on our hands! A rogue operative, possibly a Bully Cat, has infiltrated the perimeter with a confidence level of {confidence_score:.2f}. The intruder appears to be eyeing your fur babies' territory.

Enclosed is the reconnaissance image for your tactical assessment. We advise immediate action to secure the premises and ensure the safety of the fluff squad.

Stay vigilant, stay furry.

Over and out,
Your Security System 🛡️
    """

    # Create Email message
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = SENDER_EMAIL

    # Adding the email body to the MIMEText object
    msg.attach(MIMEText(body, "plain"))

    # Adding the image as an attachment to this email
    try:
        with open(image_path, 'rb') as attachment:
            mime_base = MIMEBase("application", "octet-stream")
            mime_base.set_payload(attachment.read())
            encoders.encode_base64(mime_base)
            mime_base.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
            msg.attach(mime_base)

    except:
        print(f"Error: The image file at '{image_path}' was not found.")

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls() # Establishes a connection to the SMTP server
            server.login(SENDER_EMAIL, SENDER_PASSWORD) # Authentication with the SMTP server
            server.sendmail(SENDER_EMAIL, SENDER_EMAIL, msg.as_string()) 
            print("Email notification sent successfully.")

        print("Email notification sent successfully")

    except:
        print("Failed to send email notification")