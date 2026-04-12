import asyncio
import nest_asyncio
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.google_search_tool import google_search
from google.genai import types

# Define variables required for Session setup and Agent execution
APP_NAME = "google_search_agent"
USER_ID = "user1234"
SESSION_ID = "1234"

# Define Agent with access to search tool
root_agent = Agent(
    name="basic_search_agent",
    model="gemini-2.0-flash-exp",
    description="Agent to answer questions using Google Search.",
    instruction="I can answer your questions by searching the internet. Just ask me anything!",
    tools=[
        google_search
    ],  # Google Search is a pre-built tool to perform Google searches.
)

# Agent Interaction
async def call_agent(query):
    """
    Helper function to call the agent with a query.
    """
    session_service = InMemorySessionService()
    # Ensure session is created before running
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    
    runner = Runner(
        agent=root_agent, 
        app_name=APP_NAME, 
        session_service=session_service
    )
    
    content = types.Content(role="user", parts=[types.Part(text=query)])
    
    # runner.run is likely a generator or returns an iterable of events
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    
    for event in events:
        if event.is_final_response():
            # Check if event.content exists and has parts
            if hasattr(event, 'content') and event.content.parts:
                final_response = event.content.parts[0].text
                print("Agent Response: ", final_response)
            else:
                print("Agent responded but no content was found.")

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(call_agent("what's the latest ai news?"))
