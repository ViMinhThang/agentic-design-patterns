from typing import Any, TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

class RequestInput(TypedDict):
    request: str

class CoordinatorOutput(TypedDict):
    decision: str
    request: Any
    output: Any

# --- Configuration ---
# Ensure your API key environment variable is set (GOOGLE_API_KEY)
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    print(f"Language model initialized: {llm.model}")
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None

# --- Define Simulated Sub-Agent Handlers ---
def booking_handler(request: str) -> str:
    """Simulates the Booking Agent handling a request"""
    print("\n--- DELEGATING TO BOOKING HANDLER ---")
    return f"Booking Handler processed request: '{request}'. Result: Simulated booking action"

def info_handler(request: str) -> str:
    """Simulates the Info Agent handling a request"""
    print("\n--- DELEGATING TO INFO HANDLER ---")
    return f"Info Handler processed request: '{request}'. Result: Simulated information retrieval."

def unclear_handler(request: str) -> str:
    """Handles requests that couldn't be delegated."""
    print("\n--- HANDLING UNCLEAR REQUEST ---")
    return f"Coordinator could not delegate request: '{request}'. Please clarify"

# --- Define Coordinator Router Chain ---
coordinator_router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Analyze the user's request and determine which specialist handler should process it.
            - If the request is related to booking flights or hotels, output 'booker'.
            - For all other general information questions, output 'info'.
            - If the request is unclear or doesn't fit either category, output 'unclear'.
            ONLY output one word: 'booker', 'info', or 'unclear'.""",
        ),
        ("user", "{request}"),
    ]
)

coordinator_router_chain = None
if llm is not None:
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

# --- Define the Delegation Logic ---
# Create the RunnableBranch.
delegation_branch = RunnableBranch(
    (lambda x: x["decision"].strip().lower() == "booker", lambda x: {"output": booking_handler(x["request"]["request"])}),
    (lambda x: x["decision"].strip().lower() == "info", lambda x: {"output": info_handler(x["request"]["request"])}),
    lambda x: {"output": unclear_handler(x["request"]["request"])}
)

# Combine the router chain and the delegation branch into a single runnable
coordinator_agent = None
if coordinator_router_chain:
    coordinator_agent = (
        {"decision": coordinator_router_chain, "request": RunnablePassthrough()}
        | delegation_branch
        | (lambda x: x["output"])
    )

# Example Usage ---
def main():
    if not llm or not coordinator_router_chain or not coordinator_agent:
        print("\nSkipping execution due to initialization failure")
        return
        
    print("--- Running with a booking request ---")
    request_a = "Book me a flight to London"
    result_a = coordinator_agent.invoke({"request": request_a})
    print(f"Final Result A: {result_a}")
    
    print("\n--- Running with an info request ---")
    request_b = "What is the capital of Italy?"
    result_b = coordinator_agent.invoke({"request": request_b})
    print(f"Final Result B: {result_b}")
    
    print("\n--- Running with an unclear request ---")
    request_c = "Tell me about quantum physics."
    result_c = coordinator_agent.invoke({"request": request_c})
    print(f"Final Result C: {result_c}")

if __name__ == "__main__":
    main()
