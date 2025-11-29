from __future__ import annotations

import sys
import os
from rag_pipeline import rag_tool
from tools import booking_persistence_tool, email_tool_with_pdf # MODIFIED NAME
from admin_dashboard import render_admin_dashboard

# --- Add project root to sys.path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import streamlit as st

# IMPORTS
from config import load_config
from chat_logic import detect_intent, store_message
from rag_pipeline import RAGStore, RAGConfig, build_rag_store_from_uploads
from rag_pipeline import rag_tool 
from tools import booking_persistence_tool, email_tool
from admin_dashboard import render_admin_dashboard

from booking_flow import (
    BookingState,
    get_missing_fields,
    generate_confirmation_text,
    update_state_from_message,
    next_question_for_missing_field,
)


def _init_app_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "booking_state" not in st.session_state:
        st.session_state.booking_state = BookingState()
    if "rag_store" not in st.session_state:
        st.session_state.rag_store = None
    if "rag_chunks" not in st.session_state:
        st.session_state.rag_chunks = []


def main():
    st.set_page_config(
        page_title="AI Hotel Booking Assistant",
        page_icon="🏨",
        layout="wide",
    )

    cfg = load_config()
    _init_app_state()

    menu = st.sidebar.radio("Navigation", ["Chat Assistant", "Admin Dashboard"])

    if menu == "Chat Assistant":
        run_chat_assistant(cfg)
    else:
        render_admin_dashboard()


def run_chat_assistant(cfg):
    st.title("🏨 AI Hotel Booking Assistant")

    st.subheader("Upload Hotel PDFs for RAG")
    uploaded_files = st.file_uploader(
        "Upload one or more hotel-related PDFs (policies, room details, etc.)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Build Knowledge Base from PDFs"):
            with st.spinner("Processing and indexing PDFs..."):
                rag_store, chunks = build_rag_store_from_uploads(
                    uploaded_files, RAGConfig()
                )
                st.session_state.rag_store = rag_store
                st.session_state.rag_chunks = chunks
            st.success(
                f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s)."
            )

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("How can I help you with your hotel stay today?")
    if not user_input:
        return

    # --- UI FIX: Display User Message Immediately ---
    # This fixes the "late" appearance of the user prompt
    with st.chat_message("user"):
        st.write(user_input)
    
    store_message(st.session_state.messages, "user", user_input)

    # --- INTELLIGENT ROUTING LOGIC ---
    
    detected_intent = detect_intent(user_input)
    final_intent = detected_intent
    
    # Keyword list to force RAG if the intent detector misses it
    rag_keywords = [
        "price", "cost", "how much", "rate", 
        "wifi", "internet", "pool", "gym", "spa", "parking",
        "check-in", "check-out", "policy", "refund", "cancel",
        "breakfast", "food", "restaurant", "location", "near"
    ]
    
    # 1. Global RAG Override (Highest Priority)
    # If the user asks about price/amenities, ALWAYS answer it, 
    # regardless of whether a booking is active or what the intent detector says.
    if any(kw in user_input.lower() for kw in rag_keywords):
        final_intent = "faq_rag"

    # 2. Active Booking Logic
    elif st.session_state.booking_state.active:
        if "cancel" in user_input.lower():
            final_intent = "booking"
        elif detected_intent == "faq_rag": 
             # Trust detector if it found RAG intent even without keywords
             final_intent = "faq_rag"
        else:
            # Otherwise, assume the user is answering the booking question
            final_intent = "booking"

    # Dispatch
    if final_intent == "booking":
        handle_booking_intent(cfg, user_input)
    elif final_intent == "faq_rag":
        handle_faq_intent(user_input)
    elif final_intent == "small_talk":
        respond("Hello! I can help you book rooms or answer questions about the hotel.")
    else:
        respond(
            "I’m not sure I understood. "
            "Are you trying to make a hotel booking or asking about hotel details?"
        )


def handle_booking_intent(cfg, user_message: str):
    state: BookingState = st.session_state.booking_state
    
    # Activate booking mode if not already active
    if not state.active:
        state.active = True
        st.session_state.booking_state = state

    lower_msg = user_message.strip().lower()

    # --- Cancellation Check ---
    if "cancel" in lower_msg:
        respond("Booking cancelled. Let me know if you'd like to start again.")
        st.session_state.booking_state = BookingState() # Reset and deactivate
        return

    # --- Confirmation Phase ---
    if state.awaiting_confirmation:
        if "confirm" in lower_msg or lower_msg in ("yes", "yes, confirm"):
            payload = state.to_payload()
            result = booking_persistence_tool(cfg, payload)

            if not result["success"]:
                respond(f"Error saving booking: {result['error']}")
                st.session_state.booking_state = BookingState()
                return

            booking_id = result["booking_id"]
    
            full_payload = payload.copy() 
            full_payload["ID"] = booking_id

            email_body = (
                "Your hotel booking is confirmed.\n\n"
                f"Booking ID: {booking_id}\n\n"
                "Please find your complete booking details attached as a PDF." # Simplified body
            )

            # --- CALL NEW FUNCTION ---
            email_result = email_tool_with_pdf( # MODIFIED FUNCTION CALL
                cfg,
                to_email=state.email,
                subject="Hotel Booking Confirmation (Attached PDF)",
                body_text=email_body, # Renamed body to body_text
                booking_payload=full_payload, # PASS THE FULL PAYLOAD
            )

            if not email_result["success"]:
                respond(
                    f"Booking confirmed (ID {booking_id}) but email failed: {email_result['error']}"
                )
            else:
                respond(
                    f"🎉 Booking confirmed! ID: {booking_id}. "
                    "A confirmation email has been sent."
                )

            st.session_state.booking_state = BookingState()
            return

        respond("Type 'confirm' to finalize or 'cancel' to stop.")
        return

    # --- Data Collection Phase ---
    # Update state with new info
    state = update_state_from_message(user_message, state)
    st.session_state.booking_state = state

    # Check for validation errors
    if state.errors:
        field, msg = next(iter(state.errors.items()))
        respond(msg)
        return

    # Check what is still missing
    missing = get_missing_fields(state)

    if missing:
        next_field = missing[0]
        respond(next_question_for_missing_field(next_field))
        return

    # If nothing missing, ask for confirmation
    summary = generate_confirmation_text(state)
    state.awaiting_confirmation = True
    st.session_state.booking_state = state

    respond(
        "Here are your booking details:\n\n"
        f"{summary}\n"
        "Type **'confirm'** to finalize or **'cancel'**."
    )


def handle_faq_intent(user_message: str):
    store: RAGStore = st.session_state.rag_store

    if store is None or store.size == 0:
        respond(
            "No hotel documents indexed yet. Upload PDFs and click "
            "'Build Knowledge Base', then ask your question again."
        )
    else:
        # Note: We do NOT deactivate booking state here. 
        # This allows the user to ask a question and then resume booking.
        respond(rag_tool(store, user_message))


def respond(text: str):
    store_message(st.session_state.messages, "assistant", text)
    with st.chat_message("assistant"):
        st.write(text)


if __name__ == "__main__":
    main()