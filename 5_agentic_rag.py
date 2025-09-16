import os
import re
import pandas as pd
import json
from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Load API key from another file for security
from KEYS import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Working directories and parameters
FAISS_PATH = r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Output Data\faiss_index"
RESULTS_FOLDER = r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\results"
EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVAL_K = 15 # Number of chunks to retrieve

# Create results folder if it doesn't exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Innovation ideas to evaluate
IDEAS = [
    {
        "idea_id": "A01",
        "company": "A",
        "title": "LitUs",
        "description": "Irrigation-as-a-Service, providing farmers with fully managed, high-tech irrigation systems for a fixed subscription fee to increase yields and reduce upfront costs"
    },
    {
        "idea_id": "A02", 
        "company": "A",
        "title": "Yaku",
        "description": "A water purification and monitoring system designed for hospitals and institutions, ensuring real-time control of water quality to protect vulnerable populations and provide reliable access to safe drinking water"
    },
    {
        "idea_id": "A03",
        "company": "A",
        "title": "Gray Treat - Gray Run",
        "description": "A gray water recycling system that reduces water waste by treating and reusing household wastewater, helping residential and hotel developments lower consumption and support sustainability goals"
    },
    {
        "idea_id": "A04",
        "company": "A",
        "title": "NetFog",
        "description": "Using net systems that extract water from the atmosphere to provide drinking water in arid regions"
    },
    {
        "idea_id": "A05",
        "company": "A",
        "title": "ReloadStorm",
        "description": "A rainwater harvesting system that captures, filters, and stores stormwater for household use, reducing dependence on drinking water supplies and promoting environmental sustainability"
    },
    {
        "idea_id": "A06",
        "company": "A",
        "title": "Rotoflex",
        "description": "Introduces flexible DWV (drainage, waste, and vent) pipe fittings that make installations and repairs in tight spaces quicker and more reliable, reducing leaks and on-site improvisation"
    },
    {
        "idea_id": "B01",
        "company": "B", 
        "title": "SmartOne",
        "description": "An aerodynamic fender skirt system with integrated thermal management that reduces truck fuel consumption by up to 4% while maintaining tire safety and brake performance, enabling fleets to cut costs and emissions, with operators benefiting from fast ROI, regulatory compliance, and improved vehicle reliability"
    },
    {
        "idea_id": "B02",
        "company": "B",
        "title": "Longevity", 
        "description": "Uses advanced polyurethane, rubber blends, and Tweel-inspired designs to create longboard wheels that last significantly longer while maintaining grip, performance, and ride quality"
    },
    {
        "idea_id": "B03",
        "company": "B",
        "title": "My Mechanic", 
        "description": "A digital platform that connects consumers and automotive retailers by enabling online/app-based scheduling with integrated reviews and ratings, while retailers manage availability and appointments through a simple subscription model"
    },
    {
        "idea_id": "B04",
        "company": "B",
        "title": "Tire Skins", 
        "description": "A customizable tire sidewall solution that gives young drivers affordable and stylish self-expression through adhesion and printing breakthroughs with renewable materials, while consumers enjoy personalization and quick application, and operators benefit from scalable pricing, repeat purchases, and brand strength"
    },
    {
        "idea_id": "B05",
        "company": "B",
        "title": "Foamy", 
        "description": "A compact electric mobility solution that provides commuters and fleets with a fun, eco-friendly, and customizable alternative to traditional cars, using lightweight recyclable materials and optimized battery capacity, while consumers enjoy affordability and style, and operators benefit from flexible sales, leasing, and upgrade options"
    },
    {
        "idea_id": "B06",
        "company": "B",
        "title": "Camber One", 
        "description": "Alignment device for trailer axles that measures and adjusts camber to reduce inside shoulder tire wear, extending tire life, improving fuel efficiency, and lowering fleet operating costs"
    },
    {
        "idea_id": "B07",
        "company": "B",
        "title": "Mobile", 
        "description": "Mobile is a mobile tire installation and storage service where customers select and buy tires online, schedule an appointment, and have them installed at their home or workplace, with recycling and storage included"
    }
]

# Chunk tracking for analysis
class ChunkTracker:
    def __init__(self):
        self.retrievals = []

    # Add a retrieval record
    def add(self, query: str, docs: List[Document], retrieval_type: str):
        self.retrievals.append({
            "query": query, # e.g., "Phase1 query" or "Phase2 query"
            "type": retrieval_type, # e.g., "Phase1" or "Phase2"
            "num_chunks": len(docs), # Number of chunks retrieved
            "chunks": [
                {
                    "chunk_file": d.metadata.get("chunk_file"), # File path of the chunk
                    "company_id": d.metadata.get("company_id"), # Company ID from metadata
                    "content_preview": d.page_content[:200] # First 200 characters of content
                }
                for d in docs
            ]
        })

# Phase 1 prompt builder
def phase1_prompt(idea_id, company, title, description, context):
    return f"""You are an Innovation Evaluation Assistant. Your task is to evaluate innovation ideas. 
    Act as a professional evaluator in a corporate setting.

INNOVATION:
- ID: {idea_id}
- Company: {company}
- Title: {title}
- Description: {description}

Scoring Rubric (1–10 scale):
- Uniqueness: 
  1–3 = Very similar to existing solutions
  4–6 = Some differentiation but overlaps with existing solutions
  7–10 = Highly novel or not widely seen in the market
- Market Potential: 
  1–3 = Very small/niche market
  4–6 = Moderate adoption potential, with barriers
  7–10 = Large market with strong growth opportunities
- Hotness: 
  1–3 = Low urgency, not a major pain-point
  4–6 = Some urgency, demand emerging
  7–10 = Strong urgency, pressing need

Task: Based on the following context, evaluate the innovation.

Then, select the single most useful additional context for a second retrieval 
and explain in 1–2 sentences why you chose it.

Options:
A) Market research – industry trends and competitor insights
B) Technical feasibility – technology constraints, innovation maturity
C) Economic impact – cost-benefit, ROI, financial implications
D) Regulatory research – compliance, laws, standards
E) Consumer behavior – user preferences, adoption patterns

Answer in this structured format:
Uniqueness: <score> 
Market Potential: <score>
Hotness: <score>
Additional Context Choice: <A, B, C, D, or E>

Context:
{context}
"""

# Phase 2 prompt builder
def phase2_prompt(phase1_output, additional_context):
    return f"""You are an Innovation Evaluation Assistant. Your task is to evaluate innovation ideas. 
    Act as a professional evaluator in a corporate setting.

PHASE 1 ASSESSMENT:
{phase1_output}

ADDITIONAL CONTEXT:
{additional_context}

Scoring Rubric (1–10 scale):
- Uniqueness: 
  1–3 = Very similar to existing solutions
  4–6 = Some differentiation but overlaps with existing solutions
  7–10 = Highly novel or not widely seen in the market
- Market Potential: 
  1–3 = Very small/niche market
  4–6 = Moderate adoption potential, with barriers
  7–10 = Large market with strong growth opportunities
- Hotness: 
  1–3 = Low urgency, not a major pain-point
  4–6 = Some urgency, demand emerging
  7–10 = Strong urgency, pressing need

Task: Based on the following context, evaluate the innovation.

Answer in this structured format:
Uniqueness: <score> 
Market Potential: <score>
Hotness: <score>

"""

# Extract phase 2 choice from phase 1 output for further analysis
def extract_phase2_choice(text: str) -> str:

    text = text.upper() # Normalize to uppercase for consistent matching
    match = re.search(r"\b([A-E])\b", text) # Match single letters A to E
    if match:
        return match.group(1)  
    else:
        return "A" 

# Main scoring function
def run_scoring(ideas: List[Dict], faiss_path: str, bootstrap_runs: int = 10, results_folder: str = RESULTS_FOLDER):
    # Ensure results folder exists
    os.makedirs(results_folder, exist_ok=True)
    csv_file = os.path.join(results_folder, "results.csv") # File to save aggregated results
    json_file = os.path.join(results_folder, "logs.json") # File to save detailed logs with retrieved chunk info

    # Load FAISS vector store, set up retriever and LLM
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.5) # Mixed temperature for balance

    results, logs = [], []

    # Context search queries for phase 2 retrieval
    CONTEXT_SEARCH_QUERIES = {
        'A': 'competitive analysis',
        'B': 'technical feasibility research',
        'C': 'economic impact studies',
        'D': 'regulatory research',
        'E': 'consumer behavior studies'
    }

    for idea in ideas:
        for run in range(bootstrap_runs):
            tracker = ChunkTracker()

            # Append company to query to increase retrieval relevance
            query_phase1 = f"{idea['title']} {idea['description']} company {idea['company']}"
            docs_phase1 = retriever.get_relevant_documents(query_phase1)
            
            # Filter by company_id in metadata
            docs_phase1 = [d for d in docs_phase1 if getattr(d, "metadata", {}).get("company_id") == idea["company"]]
            tracker.add("Phase1 query", docs_phase1, "phase1")
            context_phase1 = "\n\n".join([d.page_content for d in docs_phase1])

            # Phase 1 LLM
            prompt1 = phase1_prompt(idea["idea_id"], idea["company"], idea["title"], idea["description"], context_phase1)
            phase1_response_obj = llm.invoke(prompt1)
            phase1_response = getattr(phase1_response_obj, "content", str(phase1_response_obj))

            # Phase 2 retrieval
            choice = extract_phase2_choice(phase1_response)
            query_phase2 = f"{idea['title']} {idea['description']} {CONTEXT_SEARCH_QUERIES[choice]} company {idea['company']}"
            docs_phase2 = retriever.get_relevant_documents(query_phase2)
            docs_phase2 = [d for d in docs_phase2 if getattr(d, "metadata", {}).get("company_id") == idea["company"]]
            tracker.add("Phase2 query", docs_phase2, "phase2")
            context_phase2 = "\n\n".join([d.page_content for d in docs_phase2])

            # Phase 2 LLM
            prompt2 = phase2_prompt(phase1_response, context_phase2)
            phase2_response_obj = llm.invoke(prompt2)
            phase2_response = getattr(phase2_response_obj, "content", str(phase2_response_obj))

            # Parse scores
            scores = {}
            for key in ["Uniqueness", "Market Potential", "Hotness"]:
                try:
                    line = next(l for l in phase2_response.splitlines() if key.lower() in l.lower())
                    number = int(''.join(filter(str.isdigit, line.split(":")[1].strip())))
                    scores[key] = max(1, min(10, number))
                except:
                    scores[key] = 1 # Default to 1 if parsing fails, but it doesn't happen according to results

            total_chunks = tracker.retrievals[0]["num_chunks"] + tracker.retrievals[1]["num_chunks"]

            # Append results and logs
            results.append({
                "idea_id": idea["idea_id"],
                "company": idea["company"],
                "title": idea["title"],
                "run": run+1,
                "Uniqueness": scores["Uniqueness"],
                "Market Potential": scores["Market Potential"],
                "Hotness": scores["Hotness"],
                "total_chunks_retrieved": total_chunks,
                "phase2_choice": choice
            })

            logs.append({
                "idea_id": idea["idea_id"],
                "run": run+1,
                "phase1_response": phase1_response,
                "phase2_response": phase2_response,
                "total_chunks_retrieved": total_chunks,
                "phase2_choice": choice,
                "retrieved_chunks": tracker.retrievals 
})

            print(f"Run {run+1} for {idea['idea_id']} done. Total chunks: {total_chunks}")

    # Save results
    pd.DataFrame(results).to_csv(csv_file, index=False)
    with open(json_file, "w") as f:
        json.dump(logs, f, indent=2)
    return results, logs


# Execute scoring
run_scoring(IDEAS, faiss_path=FAISS_PATH, bootstrap_runs=10) # 10 runs per innovation for bootstrap analysis