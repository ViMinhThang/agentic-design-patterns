import time
from google.adk.sessions import DatabaseSessionService, InMemorySessionService, VertexAiSessionService, Session
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.genai.types import Content, Part
from google.adk.tools.tool_context import ToolContext, InvocationContext
from google.adk.memory import InMemoryMemoryService, VertexAiRagMemoryService

# Example: Using InMemorySessionService
# This is suitable for local development and testing where data
# persistence across application restarts is not required.
session_service = InMemorySessionService()

# Example: Using DatabaseSessionService
# This is suitable for production or development requiring persistent storage.
# You need to configure a database URL (e.g., for SQLite, PostgreSQL, etc.).
# Requires: pip install google-adk[sqlalchemy] and a database driver (e.g., psycopg2 for PostgreSQL)
# db_url = "sqlite:///./my_agent_data.db"
# session_service = DatabaseSessionService(db_url=db_url)

# Example: Using VertexAiSessionService
# This is suitable for scalable production on Google Cloud Platform.
PROJECT_ID = "your-gcp-project-id" # Replace with your GCP project ID
LOCATION = "us-central1" # Replace with your desired GCP location
REASONING_ENGINE_APP_NAME = "projects/your-gcp-project-id/locations/us-central1/reasoningEngines/your-engine-id" 
# session_service = VertexAiSessionService(project=PROJECT_ID, location=LOCATION)

# Define an LlmAgent with an output_key.
greeting_agent = LlmAgent(
    name="Greeter",
    model="gemini-2.0-flash",
    instruction="Generate a short, friendly greeting.",
    output_key="last_greeting"
)

# --- Setup Runner and Session ---
app_name, user_id, session_id = "state_app", "user1", "session1"
session_service = InMemorySessionService()
runner = Runner(
    agent=greeting_agent,
    app_name=app_name,
    session_service=session_service
)
session = session_service.create_session(
    app_name=app_name,
    user_id=user_id,
    session_id=session_id
)
print(f"Initial state: {session.state}")

# --- Run the Agent ---
user_message = Content(parts=[Part(text="Hello")])
print("\n--- Running the agent ---")
for event in runner.run(
    user_id=user_id,
    session_id=session_id,
    new_message=user_message
):
    if event.is_final_response():
        print("Agent responded.")

# --- Check Updated State ---
# Correctly check the state *after* the runner has finished processing all events.
updated_session = session_service.get_session(app_name, user_id, session_id)
print(f"\nState after agent run: {updated_session.state}")

# --- Define the Recommended Tool-Based Approach ---
def log_user_login(tool_context: ToolContext) -> dict:
    """
    Updates the session state upon a user login event.
    """
    # Access the state directly through the provided context.
    state = tool_context.state
    # Get current values or defaults, then update the state.
    login_count = state.get("user:login_count", 0) + 1
    state["user:login_count"] = login_count
    state["task_status"] = "active"
    state["user:last_login_ts"] = time.time()
    state["temp:validation_needed"] = True
    print("State updated from within the `log_user_login` tool.")
    return {
        "status": "success",
        "message": f"User login tracked. Total logins: {login_count}."
    }

# --- Demonstration of Tool Usage ---
# Setup for tool demonstration
session_service_tool = InMemorySessionService()
app_name_tool, user_id_tool, session_id_tool = "state_app_tool", "user3", "session3"
session_tool = session_service_tool.create_session(
    app_name=app_name_tool,
    user_id=user_id_tool,
    session_id=session_id_tool,
    state={"user:login_count": 0, "task_status": "idle"}
)
print(f"\nInitial state (tool demo): {session_tool.state}")

# Simulate a tool call
mock_context = ToolContext(
    invocation_context=InvocationContext(
        app_name=app_name_tool, 
        user_id=user_id_tool, 
        session_id=session_id_tool,
        session=session_tool, 
        session_service=session_service_tool
    )
)

log_user_login(mock_context)

# Check the updated state
updated_session_tool = session_service_tool.get_session(app_name_tool, user_id_tool, session_id_tool)
print(f"State after tool execution: {updated_session_tool.state}")

# Example: Using InMemoryMemoryService
memory_service = InMemoryMemoryService()

# Example: Using VertexAiRagMemoryService
RAG_CORPUS_RESOURCE_NAME = "projects/your-gcp-project-id/locations/us-central1/ragCorpora/your-corpus-id"
SIMILARITY_TOP_K = 5
VECTOR_DISTANCE_THRESHOLD = 0.7
# memory_service = VertexAiRagMemoryService(
#     rag_corpus=RAG_CORPUS_RESOURCE_NAME,
#     similarity_top_k=SIMILARITY_TOP_K,
#     vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD
# )
