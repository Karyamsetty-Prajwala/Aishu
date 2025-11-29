# app/tools.py

# app/tools.py

from typing import Dict, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart # NEW
from email.mime.base import MIMEBase # NEW
from email import encoders # NEW
import io # NEW
from fpdf import FPDF # NEW
import smtplib
import traceback
from datetime import datetime

from config import AppConfig
from db.database import get_supabase_client


# --- BOOKING PERSISTENCE TOOL ----------------------------------------------

def booking_persistence_tool(cfg, booking_payload):
    supabase = get_supabase_client()

    email = booking_payload["email"]

    customer_lookup = (
        supabase.table("customers").select("*").eq("email", email).execute()
    )

    if len(customer_lookup.data) == 0:
        customer_insert = (
            supabase.table("customers")
            .insert(
                {
                    "name": booking_payload["customer_name"],
                    "email": email,
                    "phone": booking_payload.get("phone"),
                }
            )
            .execute()
        )
        customer_id = customer_insert.data[0]["customer_id"]
    else:
        customer_id = customer_lookup.data[0]["customer_id"]

    booking_insert = (
        supabase.table("bookings")
        .insert(
            {
                "customer_id": customer_id,
                "booking_type": booking_payload["booking_type"],
                "date": str(booking_payload["date"]),
                "time": str(booking_payload["time"]),
                "status": "confirmed",
                "created_at": datetime.utcnow().isoformat(),
            }
        )
        .execute()
    )

    booking_id = booking_insert.data[0]["id"]

    return {
        "success": True,
        "booking_id": booking_id,
        "customer_id": customer_id,
        "error": None,
    }


# --- EMAIL TOOL -------------------------------------------------------------

def email_tool(cfg: AppConfig, to_email: str, subject: str, body: str) -> Dict[str, Any]:
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = f"{cfg.email.from_name} <{cfg.email.from_email}>"
    msg["To"] = to_email

    try:
        with smtplib.SMTP(cfg.email.smtp_host, cfg.email.smtp_port) as server:
            server.starttls()
            server.login(cfg.email.smtp_user, cfg.email.smtp_password)
            server.send_message(msg)
        return {"success": True, "error": None}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# --- OPTIONAL WEB SEARCH TOOL ----------------------------------------------

def web_search_tool(query: str) -> Dict[str, Any]:
    return {
        "success": False,
        "results": [],
        "error": "Web search tool not implemented.",
    }
def generate_booking_pdf(booking_details: dict) -> bytes:
    """Generates a booking confirmation PDF and returns its bytes."""
    
    # 1. Initialize PDF
    pdf = FPDF()
    pdf.add_page()
    
    # 2. Set Fonts and Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Hotel Booking Confirmation", 0, 1, "C")
    
    # Add separator
    pdf.line(10, 20, 200, 20)
    pdf.cell(0, 5, "", 0, 1)

    # 3. Add Booking Details
    pdf.set_font("Arial", "", 12)
    
    # Loop through the details and add them to the PDF
    # Note: Use the keys available in the payload passed from main.py
    details_to_show = {
        "Confirmation ID": booking_details.get("ID", "N/A"),
        "Guest Name": booking_details.get("customer_name", "N/A"),
        "Email": booking_details.get("email", "N/A"),
        "Phone": booking_details.get("phone", "N/A"),
        "Room Type": booking_details.get("booking_type", "N/A"),
        "Check-in Date": str(booking_details.get("date", "N/A")),
        "Arrival Time": str(booking_details.get("time", "N/A")),
    }

    for label, value in details_to_show.items():
        pdf.cell(50, 8, f"{label}:", 0)
        pdf.cell(0, 8, str(value), 0, 1)
        
    pdf.cell(0, 10, "", 0, 1) # Space

    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 5, "Thank you for booking with us.", 0, 1, "C")

    # 4. Return PDF as bytes using a buffer
    # FPDF's dest='S' returns the document as a string (bytes in Python 3.x),
    # which is exactly what we need for the attachment.
    pdf_bytes = pdf.output(dest='S')
    
    return pdf_bytes


# --- EMAIL TOOL (MODIFIED) ---------------------------------------------------

# Rename email_tool to email_tool_with_pdf to make its purpose clear
def email_tool_with_pdf(cfg: AppConfig, to_email: str, subject: str, body_text: str, booking_payload: Dict[str, Any]) -> Dict[str, Any]:
    
    # 1. Generate PDF
    pdf_data = generate_booking_pdf(booking_payload)
    
    # 2. Create MIMEMultipart message for attachments
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = f"{cfg.email.from_name} <{cfg.email.from_email}>"
    msg["To"] = to_email

    # Attach a text body
    msg.attach(MIMEText(body_text, "plain"))

    # 3. Create the PDF attachment
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(pdf_data)
    encoders.encode_base64(part)
    
    # Set the filename using the booking ID
    filename = f"Booking_{booking_payload.get('ID', 'Confirmation')}.pdf"
    part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
    
    msg.attach(part)

    # 4. Send the email
    try:
        with smtplib.SMTP(cfg.email.smtp_host, cfg.email.smtp_port) as server:
            server.starttls()
            server.login(cfg.email.smtp_user, cfg.email.smtp_password)
            server.send_message(msg)
        return {"success": True, "error": None}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# --- OPTIONAL WEB SEARCH TOOL (Existing code remains here) ---
# ... (web_search_tool function)
