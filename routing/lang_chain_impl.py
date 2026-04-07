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


# --- Configuration ---
# Ensure your API key environment variable is set (GOOGLE_API_KEY)
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    print(f"Language model initialzied: {llm.model}")
except Exception as e:
    print(f"Error initialzing languague model: {e}")
    llm = None


# --- Define Simulated Sub-Agent Handlers (equivalent to ADK sub_agents) ---
def booking_handler(request: str) -> str:
    """Simulates the Booking Agent handling a request"""
    print("\n--- DELEGATING TO BOOKING HANDLER ---")
    return f"Booking Handler processed request : '{request}'. Result: Simulated booking action"


def info_hanlder(request: str) -> str:
    """Simulates the Info Agent handling a request"""
    print("\n--- DELEGATING TO INFO HANDLER ---")
    return f"Info Handler processed request: '{request}' .Result: Simulated infomation retrieval."


def unclear_handler(request: str) -> str:
    """Handles requiests that couldn't be delegated."""
    print("\n--- HANDLING UNCLEAR REQUEST ---")
    return f"Coordinator could not delegate request: '{request}'. Please clarify"


# --- Define Coordinator Router Chain (equivalent to ADK coordinator's instruction)
coordinator_router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Analyze the user's request and determine which specialist handler should process
            - If the request is related to booking flights or hotels, output 'booker'.
            - For all other general information questions, output 'info'.
            - If the request is unclear or doesn't fit either category, output 'unclear'.
            ONLY output one word: 'booker','info', or 'unclear'.""",
        ),
        ("user", "{request}"),
    ]
)
coordinator_router_chain = None
if llm is not None:
    assert llm is not None
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

# --- Define the Delegation Logic (equivalent to ADK's Auto-Flow base on sub agents) ---
# Use RunnableBranch to route based on the router chain's output.

# Define the branches for the RunnableBranch
branches = {
    "booker": RunnablePassthrough.assign(
        output=lambda x: booking_handler(x["request"]["request"])
    ),
    "info": RunnablePassthrough.assign(
        output=lambda x: info_hanlder(x["request"]["request"])
    ),
    "unclear": RunnablePassthrough.assign(
        output=lambda x: unclear_handler(x["request"]["request"])
    ),
}


def decision_predicate(x: CoordinatorOutput) -> bool:
    return x["decision"].strip() == "booker"


def info_predicate(x: CoordinatorOutput) -> bool:
    return x["decision"].strip() == "info"


# Create the RunnableBranch. It takes the output of the route chain
# and the routes the original input ('request') to the corresponding hanlder
delegation_branch = RunnableBranch(
    (decision_predicate, branches["booker"]),
    (info_predicate, branches["info"]),
    branches["unclear"],  # Default branch for 'unclear' or any other output
)

# Combine the router chain and the delegation branch into a single runnable
# The router chain's output ('decision) is passed along with the original input('request')
# to the delegation branch
coordinator_agent = None
if coordinator_router_chain:
    coordinator_agent = (
        {"decision": coordinator_router_chain, "request": RunnablePassthrough()}
        | delegation_branch
        | (lambda x: x["output"])
    )  # Extract the final output


# Example Usage ---
def main():
    if not llm or not coordinator_router_chain or not coordinator_agent:
        print("\nSkipping execution due to LLM initialization failure")
        return
    print("--- Running with a booking request ---")
    request_a = "Book me a flight to london"
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
