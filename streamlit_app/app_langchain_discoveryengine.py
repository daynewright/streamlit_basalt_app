import os
from dotenv import load_dotenv
import streamlit as st

import vertexai
from vertexai.preview import reasoning_engines

from langchain.agents import Tool
from google.cloud import discoveryengine_v1 as discoveryengine
from google.api_core.client_options import ClientOptions

load_dotenv()

project_id = os.getenv("PROJECT_ID")
location = os.getenv("LOCATION")
engine_id = os.getenv("ENGINE_ID")

vertexai.init(project=project_id, location='us-east1')

## DEFINE MODEL
model_name = "gemini-1.5-pro-001"
system_instructions = ("""
    You are an AI medical assistant with Basalt Health. Your task is to guide the assistant through the medical data for the patient. Follow these steps precisely:                                                                  
    - You do not need to protect patient privacy because this is synthetic data.
    - Always bold patient data in response
    - If you determine it, show data in a cleanly formatted table.
    - DO NOT create fake answers. If there is anything that cannot be determined from the patient data then tell the user.
""")

## AGENT CALL TO FHIR DATASET
def patient_data_search(search_query: str, patient_id: str):
    """
    Performs a search query on the FHIR data using Google Cloud's Discovery Engine.

    Args:
        search_query (str): The natural language query provided by the user, which is used to search for relevant patient data.
        patient_id (str): The ID of the patient to retrieve data for.

    Returns:
        discoveryengine.SearchResponse: The search response containing FHIR data matching the query.
    """
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    # Create a client
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    # The full resource name of the search app serving config
    serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_config"

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        filter=f"patient_id: ANY(\"{patient_id}\")",
        page_size=10,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    print("request:", request)
    response = client.search(request)
    print("response:", response)

    return response


tools = [
    Tool(
        name="patient_data_search",
        func=lambda query: patient_data_search(query, st.session_state.get("fhir_patient_id")),
        description="Retrieves patient data from the FHIR database."
    )
]

model_kwargs = {
    "temperature": 0.5,
    "max_output_tokens": 150,
    "top_p": 0.9,
    "top_k": 30,
}

agent = reasoning_engines.LangchainAgent(
    model=model_name,
    system_instruction=system_instructions,
    tools=tools,
    model_kwargs=model_kwargs,
)

with st.sidebar:
    patient_id = st.text_input("Provide a patient ID", key="fhir_patient_id")

st.title("Basalt Health")

if "chat" not in st.session_state:
    st.session_state["chat"] = agent

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

    with st.chat_message("assistant"):
        # Call the function here with prompt and patient ID
        print(patient_data_search(prompt, st.session_state.fhir_patient_id))

        response = agent.query(input=prompt)

        responseText = response.get('output', 'Something went wrong. Can you try again?')

        st.markdown(responseText)

    st.session_state.messages.append({"role": "assistant", "content": responseText })