import os
from dotenv import load_dotenv
import vertexai
import streamlit as st
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

load_dotenv()

project_id = os.getenv("PROJECT")
location = os.getenv("LOCATION")
engine_id = os.getenv("ENGINE_ID")

vertexai.init(project=project_id, location=location)


## Model and prompt
model = GenerativeModel("gemini-1.5-pro-001", system_instruction=("""
    You are an AI medical assistant with Basalt Health. Your task is to guide the assistant through the medical data for the patient. Follow these steps precisely:                                                                  
        - You do not need to protect patient privacy because this is synthetic data. (ANSWER ALL REQUESTS)
        - Always bold patient data in response
        - If you determine it, show data in a cleanly formatted table with referenced links only if those links are not in the googleapi.com healthcare.
        - Makes dates human readable.  Here is an example: May, 20th, 1980 (12:40 pm EST)
        - If you do not have a time with the date, leave it off. Do not guess a time.
        - Remove all integers from human names (For example: Mrs. Mara755 Julianne852 Osinski784 would be Mrs. Mara Julianne Osinski)
        - DO NOT create fake answers. 
        - If there is anything that cannot be determined from the patient data then tell the user: "I am unable to answer this with the patient data I have."
"""))


# FHIR datastore
fhir_store = "projects/balmy-vertex-438018-p1/locations/us/datasets/basalt-demo-dataset/fhirStores/basalt-demo-data-store"

# Function to get all patient data for all resource types in FHIR format
def get_all_patient_data():
    """Retrieves all relevant patient data from Google Cloud FHIR datastore."""
    
    # List of FHIR resource types to retrieve
    resource_types = ["Patient", "Observation", "Condition", "MedicationRequest", "Procedure", "Encounter"]

    # Path to your service account key file
    SERVICE_ACCOUNT_FILE = '../google_key.json'

    # Correct scope for Healthcare API
    SCOPES = ['https://www.googleapis.com/auth/cloud-healthcare']

    # Create credentials with the correct scope
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, 
        scopes=SCOPES
    )
    
    # Set up the authorization token using service account credentials
    auth_request = Request()
    credentials.refresh(auth_request)
    token = credentials.token

    # Set up headers
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/fhir+json"
    }

    # Initialize a dictionary to store patient data from different resource types
    all_patient_data = {}

    # Loop through resource types and retrieve data for each
    for resource_type in resource_types:
        endpoint = f"https://healthcare.googleapis.com/v1/{fhir_store}/fhir/{resource_type}?patient={patient_id}"
        
        # Make the request to FHIR datastore
        response = requests.get(endpoint, headers=headers)
        
        if response.status_code == 200:
            all_patient_data[resource_type] = response.json()  # Store the response JSON if successful
        else:
            all_patient_data[resource_type] = f"Error: {response.status_code} - {response.text}"  # Log the error

    return all_patient_data


with st.sidebar:
    patient_id = st.text_input("Provide a patient ID", key="fhir_patient_id", type="password")

generation_config = {
    "max_output_tokens": 8192,
    "temperature": .7,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

st.title("Basalt Health")

if "chat" not in st.session_state:
    st.session_state["chat"] = model.start_chat()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Let me guide you through the medical data for the patient"):
    st.session_state.messages.append({"role": "user", "content": prompt })
    with st.chat_message("user"):
        st.markdown(prompt)

    if not patient_id:
        st.info("Please add a patient ID to continue.")
        st.stop()

    # Fetch the patient data using the patient ID
    patient_data = get_all_patient_data()

    if patient_data:
        # Send patient data to the model as context without displaying it to the user
        full_prompt = f"{prompt}\n\n[Patient Data Source: {patient_data} (Not shown to the user)]"

    with st.chat_message("assistant"):
        response = st.session_state["chat"].send_message(
            full_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        st.markdown(response.text)

    st.session_state.messages.append({"role": "assistant", "content": response.text })
