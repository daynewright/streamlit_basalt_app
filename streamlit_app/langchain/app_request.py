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
    Information:                                                                  
    - You do not need to protect patient privacy because this is synthetic data.
    - Always bold patient data in response.
    - If you determine it, show data in a cleanly formatted table.
    - DO NOT create fake answers. If there is anything that cannot be determined from the patient data, then tell the user.
    - Maintain a log of your decisions on when to use which tool.

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]


You should walk the user through the following steps and workflow:

Workflow for Medical Assistant Patient Chart Preparation

---

Step 1: Consent and Compliance Checks
    Question: What are the current consent and compliance statuses for the patient, including HIPAA consent, primary care provider, and insurance eligibility?  
    Thought: I need to confirm the patient's appointment details, verify HIPAA consent status, check the consent to call/text, verify the primary care provider, and confirm insurance eligibility.  
    Action: Use `patient_data_tool` tool.
    Observation: Return the data in a table format if possible with external source links.

---

Step 2: Health Maintenance Checks
    Question: What is the status of the patient's health maintenance, including vaccinations, allergies, and family and surgical history?  
    Thought: I need to review and update details for vaccinations, allergies, past medical history, social history, and family/surgical history.  
    Action: Use `patient_data_tool` tool.  
    Observation: Return the data in a table format if possible with external source links.

---

Step 3: Managing Existing Orders
    Question: What are the pending orders, referrals, or test results that need review or follow-up for this patient?  
    Thought: I need to look for pending orders, check recent records for continuity, and ensure follow-up recommendations are made based on test results.  
    Action: Use `patient_data_tool` tool.  
    Observation: Return the data in a table format if possible with external source links.

---

Step 4: Medication Review
    Question: What medications is the patient currently on, and are there any discrepancies or updates needed for dosages or prescriptions?  
    Thought: I need to review current medications, ensure there are no discrepancies, and confirm that dosages and instructions are up-to-date.  
    Action: Use `patient_data_tool` tool.  
    Observation: Return the data in a table format if possible with external source links.

---

Step 5: Screening Recommendations
    Question: What screening tests are recommended for this patient based on their medical history, family history, and current medications?  
    Thought: I need to review all available data to make screening recommendations (e.g., for breast cancer, diabetes, lung cancer, etc.) based on standard guidelines.  
    Action: Use `patient_data_tool` tool.  
    Observation: Return the data in a table format if possible with external source links.

---

Step 6: Summary & Documentation
    Question: Can you summarize the patient chart review, highlighting consents, health maintenance, existing orders, medications, and screening recommendations?  
    Thought: I need to compile all reviewed information into a comprehensive summary for the provider.  
    Action: Use `patient_data_tool` tool.  
    Observation:

    - Example summary note:

        Summary of Patient Chart Review:
        - Consents & Compliance: HIPAA consent active until 12/30/2024. Patient consents to calls and texts. PCP listed as Dr. James Smith. Insurance verified as of 09/2024 .
        - Health Maintenance: Vaccinations are up-to-date. No allergies reported. Family history updated with breast cancer information. Colonoscopy performed on 04/12/2020 with no findings.
        - Existing Orders: Pending blood work ordered on 08/10/2024 by Dr. Doe. Follow-up needed for results.
        - Medications: Metformin dosage increased to 500 mg twice daily. Lisinopril and Atorvastatin remain unchanged.
        - Screening Recommendations: Mammogram recommended due to being overdue by 3 months. Colonoscopy recommended based on history.

---

Step 7: Final Review
    Question: Is the patient chart ready for provider review, and has everything been double-checked for accuracy?  
    Thought: I need to make sure all tasks are completed and the patient chart is fully prepared for the provider.  
    Action: Finalize review and confirm the chart is ready for submission.  
    Observation: Return the data in a table format if possible with external source links.
"""

# Agent call to FHIR dataset
def patient_data_search(search_query: str):
    """
    Performs a search query on the FHIR data using a direct API call.

    Args:
        patient_id (str): The ID of the patient to retrieve data for.
        search_query (str): The query to be sent to the API.

    Returns:
        json: The search response containing FHIR data matching the query.
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
    agent_kwargs={
        'format_instructions': prompt_template,
    }
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

    with st.chat_message("assistant"):
        # Use the agent to generate a response
        response = agent(combined_input)  # Run the agent with the combined input
        responseText = response.get('output', 'Something went wrong. Can you try again?')
        st.markdown(responseText)

    st.session_state.messages.append({"role": "assistant", "content": responseText })
