from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

agent = AugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona)

augmented_agent_response = agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# - What knowledge the agent likely used to answer the prompt.
# The agent used its general knowledge from the GPT-3.5-turbo model to answer the question.

# - How the system prompt specifying the persona affected the agent's response.
# The persona system prompt instructed the agent to respond as a college professor,
# which caused the response to start with "Dear students," and use a more formal tone.
