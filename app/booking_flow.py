from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, time
from typing import Optional, Dict, Any, List
import json
import re

import streamlit as st
from email_validator import validate_email as _validate_email, EmailNotValidError
import google.generativeai as genai


BOOKING_FIELDS = [
    "customer_name",
    "email",
    "phone",
    "booking_type",
    "date",
    "time",
]


@dataclass
class BookingState:
    customer_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    booking_type: Optional[str] = None
    date: Optional[date] = None
    time: Optional[time] = None
    
    active: bool = False 

    awaiting_confirmation: bool = False
    errors: Dict[str, str] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "customer_name": self.customer_name,
            "email": self.email,
            "phone": self.phone,
            "booking_type": self.booking_type,
            "date": self.date,
            "time": self.time,
        }


# ----------------- VALIDATORS ------------------------

def validate_email(email: str) -> bool:
    try:
        _validate_email(email, check_deliverability=False)
        return True
    except EmailNotValidError:
        return False


def parse_date_str(val: str) -> Optional[date]:
    try:
        return datetime.strptime(val.strip(), "%Y-%m-%d").date()
    except:
        return None


def parse_time_str(val: str) -> Optional[time]:
    if not val:
        return None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(val.strip(), fmt).time()
        except:
            continue
    return None


# ----------------- SLOT HANDLING ------------------------

def get_missing_fields(state: BookingState) -> List[str]:
    missing = []
    for f in BOOKING_FIELDS:
        if getattr(state, f, None) in (None, ""):
            missing.append(f)
    return missing


def generate_confirmation_text(state: BookingState) -> str:
    return (
        f"Name: {state.customer_name}\n"
        f"Email: {state.email}\n"
        f"Phone: {state.phone or 'N/A'}\n"
        f"Room Type: {state.booking_type}\n"
        f"Date: {state.date}\n"
        f"Time: {state.time}\n"
    )


# ----------------- GEMINI EXTRACTION ------------------------

def _configure_gemini():
    try:
        if "google" in st.secrets:
            api_key = st.secrets["google"]["api_key"]
        elif "gemini" in st.secrets:
            api_key = st.secrets["gemini"]["api_key"]
        else:
            api_key = st.secrets.get("google_api_key", "")
            
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")

def llm_extract_booking_fields(message: str, state: BookingState) -> Dict[str, Any]:
    _configure_gemini()
    
    # Use gemini-1.5-flash
    model = genai.GenerativeModel('gemini-1.5-flash')

    missing = get_missing_fields(state)
    expected_field = missing[0] if missing else "none"
    today = date.today().isoformat()

    system_prompt = (
        "You extract booking fields from user text. "
        f"CURRENT CONTEXT: The system is asking the user for: '{expected_field}'. "
        f"TODAY'S DATE: {today}. "
        "If the user provides a short answer (e.g. 'John' or 'tomorrow'), assume it refers to the requested field. "
        "Return a valid JSON object (no markdown formatting) with keys: "
        "customer_name, email, phone, booking_type, date, time. "
        "Use date format YYYY-MM-DD and time HH:MM (24-hour). "
        "If a field is missing, set it to null. "
        "Do not include ```json ... ``` wrappers, just raw JSON."
    )

    prompt = f"{system_prompt}\n\nUser Message: {message}"

    try:
        response = model.generate_content(prompt)
        content = response.text
        
        # Clean up code blocks
        content = content.replace("```json", "").replace("```", "").strip()
        
        # If model returns empty, return empty dict
        if not content:
            return {}

        return json.loads(content)

    except Exception as e:
        # --- DEBUG: Show error in UI so we know WHY it failed ---
        st.error(f"Extraction Error: {str(e)}")
        return {}


# ----------------- STATE UPDATE ------------------------

def update_state_from_message(message: str, state: BookingState) -> BookingState:
    
    state.active = True

    extracted = llm_extract_booking_fields(message, state)
    state.errors.clear()

    # --- Name ---
    name = extracted.get("customer_name")
    if name and not state.customer_name:
        state.customer_name = name.strip()

    # --- Email ---
    email = extracted.get("email")
    if email and not state.email:
        email = email.strip()
        if validate_email(email):
            state.email = email
        else:
            state.errors["email"] = "Invalid email address."

    # --- Phone ---
    phone = extracted.get("phone")
    if phone and not state.phone:
        state.phone = phone.strip()

    # --- Booking Type ---
    booking_type = extracted.get("booking_type")
    if booking_type and not state.booking_type:
        state.booking_type = booking_type.strip()

    # --- Date ---
    date_str = extracted.get("date")
    if date_str and not state.date:
        parsed = parse_date_str(date_str)
        if parsed:
            state.date = parsed
        else:
            state.errors["date"] = "Please use date format YYYY-MM-DD."

    # --- Time ---
    time_str = extracted.get("time")
    if time_str and not state.time:
        parsed = parse_time_str(time_str)
        if parsed:
            state.time = parsed
        else:
            state.errors["time"] = "Please use time format HH:MM (24h)."

    return state


# ----------------- QUESTIONS ------------------------

def next_question_for_missing_field(field_name: str) -> str:
    prompts = {
        "customer_name": "May I know the guest name?",
        "email": "What's your email address for confirmation?",
        "phone": "Your phone number? (optional)",
        "booking_type": "What type of room would you like to book?",
        "date": "What check-in date? Please use YYYY-MM-DD.",
        "time": "What arrival time? Please use HH:MM (24-hour).",
    }
    return prompts.get(field_name, f"Please provide {field_name}.")