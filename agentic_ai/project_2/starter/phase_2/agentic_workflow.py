from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent

import os
from dotenv import load_dotenv

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
with open("Product-Spec-Email-Router.txt", "r") as f:
    product_spec = f.read()

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)

action_planning_agent = ActionPlanningAgent(openai_api_key=openai_api_key, knowledge=knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "User stories MUST follow this EXACT format, one story per line:\n"
    "As a [type of user], I want [an action or feature] so that [benefit/value].\n\n"
    "RULES:\n"
    "- Every story starts with 'As a'\n"
    "- Every story contains 'I want'\n"
    "- Every story contains 'so that'\n"
    "- Every story ends with a period after the benefit\n"
    "- Do NOT use bullet points, numbered lists, or sub-items\n"
    "- Do NOT add extra sentences after the period\n"
    "- Do NOT use 'I need', 'I expect', or 'so I can' instead of 'I want' and 'so that'\n\n"
    "Example: As a customer support representative, I want automated email categorization so that I can focus on complex inquiries.\n\n"
    "Write user stories for the product spec below:\n"
    + product_spec
)

product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager
)

# Product Manager - Evaluation Agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_product_manager = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    worker_agent=product_manager_knowledge_agent,
    max_interactions=10
)


# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = (
    "Features of a product are defined by organizing similar user stories into cohesive groups. "
    "Each feature MUST be written using EXACTLY this structure:\n"
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user\n\n"
    + product_spec
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager
)

program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=(
        "The answer should be product features that follow the following structure: "
        "Feature Name: A clear, concise title that identifies the capability\n"
        "Description: A brief explanation of what the feature does and its purpose\n"
        "Key Functionality: The specific capabilities or actions the feature provides\n"
        "User Benefit: How this feature creates value for the user"
    ),
    worker_agent=program_manager_knowledge_agent,
    max_interactions=10
)


# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = (
    "Development tasks are defined by identifying what needs to be built to implement each user story. "
    "Each task MUST use EXACTLY these field names with no variation, no hyphens, and no extra text:\n"
    "Task ID: (a unique identifier, e.g. 1.1, 1.2)\n"
    "Task Title: (brief description of the development work)\n"
    "Related User Story: (reference to the parent user story)\n"
    "Description: (detailed explanation of the technical work)\n"
    "Acceptance Criteria: (specific requirements for completion)\n"
    "Estimated Effort: (time or complexity, e.g. '3 days')\n"
    "Dependencies: (tasks that must be completed first, or 'None')\n\n"
    + product_spec
)

development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."

development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=(
        "The answer must contain only engineering tasks. "
        "Every single task MUST include ALL seven of these fields:\n"
        "Task ID:\n"
        "Task Title:\n"
        "Related User Story:\n"
        "Description:\n"
        "Acceptance Criteria:\n"
        "Estimated Effort:\n"
        "Dependencies:\n"
        "Reject if ANY task is missing ANY of these seven fields. "
        "Reject if a block has Description/Acceptance Criteria/Estimated Effort/Dependencies but no Task ID, Task Title, or Related User Story. "
        "Reject if a block has Task ID/Task Title/Related User Story but no Description, Acceptance Criteria, Estimated Effort, or Dependencies. "
        "Reject if 'Effort Estimate:' is used instead of 'Estimated Effort:'."
    ),
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=10
)

# Final Assembly Agent - consolidates all workflow outputs into a structured final document
persona_final_assembler = (
    "You are a Technical Writer and Project Manager. "
    "You compile project documentation into a clean, structured format. "
    "You output only the structured content with no preambles or commentary."
)
knowledge_final_assembler = (
    "You will receive outputs from multiple planning agents. "
    "Your sole task is to compile them into a final project plan with EXACTLY three sections.\n\n"
    "SECTION RULES:\n"
    "1. User Stories: each story MUST follow this exact pattern (one per line):\n"
    "   As a [type of user], I want [an action or feature] so that [benefit/value].\n\n"
    "2. Product Features: each feature MUST use ALL FOUR of these exact field labels:\n"
    "   Feature Name: <name>\n"
    "   Description: <brief explanation>\n"
    "   Key Functionality: <specific capabilities>\n"
    "   User Benefit: <how it creates value>\n\n"
    "3. Engineering Tasks: each task MUST use ALL SEVEN of these exact field labels:\n"
    "   Task ID: <unique id>\n"
    "   Task Title: <brief title>\n"
    "   Related User Story: <story reference>\n"
    "   Description: <detailed explanation>\n"
    "   Acceptance Criteria: <specific requirements>\n"
    "   Estimated Effort: <time estimate>\n"
    "   Dependencies: <prerequisite tasks or None>\n\n"
    "Remove duplicates within each section. "
    "Output ONLY the three sections with these exact headings and nothing else."
)

final_assembly_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_final_assembler,
    knowledge=knowledge_final_assembler
)

evaluation_criteria_final_assembly = (
    "The output MUST contain exactly three sections with these EXACT headings:\n"
    "=== User Stories ===\n"
    "=== Product Features ===\n"
    "=== Engineering Tasks ===\n\n"
    "User Stories: every story must start with 'As a', contain 'I want', and contain 'so that'.\n\n"
    "Product Features: every feature block must contain ALL FOUR labels: "
    "'Feature Name:', 'Description:', 'Key Functionality:', 'User Benefit:'.\n\n"
    "Engineering Tasks: every task block must contain ALL SEVEN labels: "
    "'Task ID:', 'Task Title:', 'Related User Story:', 'Description:', "
    "'Acceptance Criteria:', 'Estimated Effort:', 'Dependencies:'.\n\n"
    "Reject if any feature is missing any of the four required labels. "
    "Reject if any task is missing any of the seven required labels. "
    "Reject if any section heading is missing or uses different text."
)

final_assembly_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona="You are a strict evaluator checking that a final project plan meets exact structural requirements.",
    evaluation_criteria=evaluation_criteria_final_assembly,
    worker_agent=final_assembly_agent,
    max_interactions=10
)

# Routing Agent
routing_agent = RoutingAgent(openai_api_key=openai_api_key, agents=[])
agents = [
    {
        "name": "Product Manager",
        "description": "Writes and identifies user stories by defining personas and customer needs. Creates story requirements from the user perspective.",
        "func": lambda x: product_manager_support_function(x)
    },
    {
        "name": "Program Manager",
        "description": "Defines product features and capabilities by organizing requirements into cohesive product releases and roadmaps.",
        "func": lambda x: program_manager_support_function(x)
    },
    {
        "name": "Development Engineer",
        "description": "Creates engineering tasks and technical implementation plans. Breaks down work into coding, testing, and deployment activities with timelines.",
        "func": lambda x: development_engineer_support_function(x)
    }
]
routing_agent.agents = agents

def product_manager_support_function(query):
    """Get user stories from the Product Manager agent and evaluate them."""
    response = product_manager_knowledge_agent.respond(query)
    result = product_manager_evaluation_agent.evaluate(response)
    return "=== User Stories ===\n\n" + result['final_response']

def program_manager_support_function(query):
    """Get product features from the Program Manager agent and evaluate them."""
    response = program_manager_knowledge_agent.respond(query)
    result = program_manager_evaluation_agent.evaluate(response)
    return "=== Product Features ===\n\n" + result['final_response']

def development_engineer_support_function(query):
    """Get development tasks from the Development Engineer agent and evaluate them."""
    response = development_engineer_knowledge_agent.respond(query)
    result = development_engineer_evaluation_agent.evaluate(response)
    return "=== Engineering Tasks ===\n\n" + result['final_response']

# Run the workflow

print("\n*** Workflow execution started ***\n")
workflow_prompt = "Create a complete development plan for this product including user stories, product features, and development tasks."
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")

workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
completed_steps = []

for step in workflow_steps:
    print(f"\nExecuting step: {step}")
    result = routing_agent.route(step)
    completed_steps.append(result)
    print(f"Result: {result}")

print("\n*** Assembling final structured output ***")
all_content = "\n\n---\n\n".join(completed_steps)
final_assembly_prompt = (
    "Below are the outputs from a project planning workflow for the Email Router system:\n\n"
    + all_content
    + "\n\n"
    "Compile the above into a FINAL STRUCTURED PROJECT PLAN with EXACTLY three sections "
    "using these EXACT headings:\n\n"
    "=== User Stories ===\n"
    "(List all unique user stories. Each MUST follow exactly: "
    "As a [type of user], I want [an action or feature] so that [benefit/value].)\n\n"
    "=== Product Features ===\n"
    "(List all unique features. Each MUST use all four exact fields:\n"
    "Feature Name: <name>\n"
    "Description: <brief explanation>\n"
    "Key Functionality: <specific capabilities>\n"
    "User Benefit: <how it creates value>)\n\n"
    "=== Engineering Tasks ===\n"
    "(List all unique tasks. Each MUST use all seven exact fields:\n"
    "Task ID: <unique id>\n"
    "Task Title: <brief title>\n"
    "Related User Story: <story reference>\n"
    "Description: <detailed explanation>\n"
    "Acceptance Criteria: <specific requirements>\n"
    "Estimated Effort: <time estimate>\n"
    "Dependencies: <prerequisite tasks or None>)\n\n"
    "Include ONLY the three sections above. No introductory text, no conclusion."
)

raw_final = final_assembly_agent.respond(final_assembly_prompt)
final_result = final_assembly_evaluation_agent.evaluate(raw_final)

print("\n*** Final output ***")
print(final_result['final_response'])
