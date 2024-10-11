import os
import json
import requests
from dotenv import load_dotenv
import streamlit as st

import vertexai
from langchain.agents import Tool, initialize_agent

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


load_dotenv()

project_id = os.getenv("PROJECT_ID")
location = os.getenv("LOCATION")
engine_id = os.getenv("ENGINE_ID")

vertexai.init(project=project_id, location='us-east1')

# Define model name and system instructions
model_name = "gemini-1.5-pro-001"
system_instructions = """
    You are an AI medical assistant with Basalt Health. Your task is to guide the assistant through the medical data for the patient. 
    Follow these steps precisely:                                                                  
    - You do not need to protect patient privacy because this is synthetic data.
    - Always bold patient data in response.
    - If you determine it, show data in a cleanly formatted table.
    - DO NOT create fake answers. If there is anything that cannot be determined from the patient data, then tell the user.
    - Maintain a log of your decisions on when to use which tool.
"""

# Agent call to FHIR dataset
def patient_data_search(search_query: str):
    """
    Performs a search query on the FHIR data using a direct API call.

    Args:
        patient_id (str): The ID of the patient to retrieve data for.
        search_query (str): The query to be sent to the API.

    Returns:
        dict: The search response containing FHIR data matching the query.
    """
    access_token = os.popen("gcloud auth print-access-token").read().strip()

    url = f"https://{location}-discoveryengine.googleapis.com/v1alpha/projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_serving_config:search"

    payload = {
        "query": search_query,
        "filter": f"patient_id: ANY(\"{patient_id}\")",
        "contentSearchSpec": {
            "snippetSpec": {
                "returnSnippet": True
            },
            "summarySpec": {
                "summaryResultCount": 1,
                "includeCitations": True,
            }
        },
        "naturalLanguageQueryUnderstandingSpec": {
            "filterExtractionCondition": "ENABLED"
        }
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        return None

    return response.json()

# Define a Langchain tool for patient data retrieval
patient_data_tool = Tool(
    name="Patient Data Search",
    func=patient_data_search,
    description="Useful when you are asked about patient spefic data. The tool already contains the patient id so it only needs the question from the user"
)

# Create the LLM
llm = ChatVertexAI(
    temperature=0,
    model_name=model_name)

# Conversational agent memory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)

# Create a prompt template
prompt_template = ChatPromptTemplate.from_template(system_instructions)

# Create the agent
agent = initialize_agent(
    agent='conversational-react-description',
    llm=llm,
    tools=[patient_data_tool],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=memory,
)

with st.sidebar:
    patient_id = st.text_input("Provide a patient ID", key="fhir_patient_id")

st.title("Basalt Health")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Let me guide you through the medical data for the patient"):
    st.session_state.messages.append({"role": "user", "content": prompt })
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.fhir_patient_id:
        st.info("Please add a patient ID to continue.")
        st.stop()

    # Combine prompt and patient ID for context
    combined_input = f"{prompt}\nPatient ID: {st.session_state.fhir_patient_id}"

    print(combined_input)

    with st.chat_message("assistant"):
        # Use the agent to generate a response
        response = agent(combined_input)  # Run the agent with the combined input
        responseText = response.get('output', 'Something went wrong. Can you try again?')
        st.markdown(responseText)

    st.session_state.messages.append({"role": "assistant", "content": responseText })
