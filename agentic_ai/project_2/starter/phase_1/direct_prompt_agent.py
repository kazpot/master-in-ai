from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

direct_agent = DirectPromptAgent(openai_api_key=openai_api_key)
direct_agent_response = direct_agent.respond(prompt=prompt)

# Print the response from the agent
print(direct_agent_response)
print("The agent used its general knowledge from the GPT-3.5-turbo language model to answer this question.")
