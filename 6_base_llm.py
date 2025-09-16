import os
import pandas as pd
import json
from typing import List, Dict
from langchain.chat_models import ChatOpenAI

# Load API key from another file for security
from KEYS import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Innovation ideas to evaluate
IDEAS = [
    {
        "idea_id": "A01",
        "company": "Company A",
        "title": "LitUs",
        "description": "Irrigation-as-a-Service, providing farmers with fully managed, high-tech irrigation systems for a fixed subscription fee to increase yields and reduce upfront costs"
    },
    {
        "idea_id": "A02", 
        "company": "Company A",
        "title": "Yaku",
        "description": "A water purification and monitoring system designed for hospitals and institutions, ensuring real-time control of water quality to protect vulnerable populations and provide reliable access to safe drinking water"
    },
    {
        "idea_id": "A03",
        "company": "Company A",
        "title": "Gray Treat - Gray Run",
        "description": "A gray water recycling system that reduces water waste by treating and reusing household wastewater, helping residential and hotel developments lower consumption and support sustainability goals"
    },
    {
        "idea_id": "A04",
        "company": "Company A",
        "title": "NetFog",
        "description": "Using net systems that extract water from the atmosphere to provide drinking water in arid regions"
    },
    {
        "idea_id": "A05",
        "company": "Company A",
        "title": "ReloadStorm",
        "description": "A rainwater harvesting system that captures, filters, and stores stormwater for household use, reducing dependence on drinking water supplies and promoting environmental sustainability"
    },
    {
        "idea_id": "A06",
        "company": "Company A",
        "title": "Rotoflex",
        "description": "Introduces flexible DWV (drainage, waste, and vent) pipe fittings that make installations and repairs in tight spaces quicker and more reliable, reducing leaks and on-site improvisation"
    },
    {
        "idea_id": "B01",
        "company": "Company B", 
        "title": "SmartOne",
        "description": "An aerodynamic fender skirt system with integrated thermal management that reduces truck fuel consumption by up to 4% while maintaining tire safety and brake performance, enabling fleets to cut costs and emissions, with operators benefiting from fast ROI, regulatory compliance, and improved vehicle reliability"
    },
    {
        "idea_id": "B02",
        "company": "Company B",
        "title": "Longevity", 
        "description": "Uses advanced polyurethane, rubber blends, and Tweel-inspired designs to create longboard wheels that last significantly longer while maintaining grip, performance, and ride quality"
    },
    {
        "idea_id": "B03",
        "company": "Company B",
        "title": "My Mechanic", 
        "description": "A digital platform that connects consumers and automotive retailers by enabling online/app-based scheduling with integrated reviews and ratings, while retailers manage availability and appointments through a simple subscription model"
    },
    {
        "idea_id": "B04",
        "company": "Company B",
        "title": "Tire Skins", 
        "description": "A customizable tire sidewall solution that gives young drivers affordable and stylish self-expression through adhesion and printing breakthroughs with renewable materials, while consumers enjoy personalization and quick application, and operators benefit from scalable pricing, repeat purchases, and brand strength"
    },
    {
        "idea_id": "B05",
        "company": "Company B",
        "title": "Foamy", 
        "description": "A compact electric mobility solution that provides commuters and fleets with a fun, eco-friendly, and customizable alternative to traditional cars, using lightweight recyclable materials and optimized battery capacity, while consumers enjoy affordability and style, and operators benefit from flexible sales, leasing, and upgrade options"
    },
    {
        "idea_id": "B06",
        "company": "Company B",
        "title": "Camber One", 
        "description": "Alignment device for trailer axles that measures and adjusts camber to reduce inside shoulder tire wear, extending tire life, improving fuel efficiency, and lowering fleet operating costs"
    },
    {
        "idea_id": "B07",
        "company": "Company B",
        "title": "Mobile", 
        "description": "Mobile is a mobile tire installation and storage service where customers select and buy tires online, schedule an appointment, and have them installed at their home or workplace, with recycling and storage included"
    }
]

# Prompt builder
def direct_scoring_prompt(idea_id: str, company: str, title: str, description: str) -> str:
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

Answer in this structured format:
Uniqueness: <score> 
Market Potential: <score>
Hotness: <score>
"""

# Parsing function
def parse_scores(text: str) -> Dict[str, int]:
    scores = {}
    for line in text.splitlines():
        for key in ["Uniqueness", "Market Potential", "Hotness"]:
            if line.startswith(f"{key}:"):
                try:
                    scores[key] = int(line.split(":")[1].strip())
                except ValueError:
                    scores[key] = None
    return scores

# Benchmarking/Base LLM function
def run_benchmark(ideas: List[Dict], bootstrap_runs=10, results_folder=r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\results_benchmark"):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.5) # Mixed temperature for balance

    results, logs = [], []

    os.makedirs(results_folder, exist_ok=True)
    csv_file = os.path.join(results_folder, "benchmark_results.csv")
    json_file = os.path.join(results_folder, "benchmark_logs.json")

    for idea in ideas:
        for run in range(bootstrap_runs):
            prompt = direct_scoring_prompt(idea["idea_id"], idea["company"], idea["title"], idea["description"])
            response_obj = llm.invoke(prompt)
            response = response_obj.content if hasattr(response_obj, "content") else str(response_obj)
            response = response.strip()

            scores = parse_scores(response)

            results.append({
                "idea_id": idea["idea_id"],
                "company": idea["company"],
                "title": idea["title"],
                "run": run + 1,
                "Uniqueness": scores.get("Uniqueness"),
                "Market Potential": scores.get("Market Potential"),
                "Hotness": scores.get("Hotness")
            })

            logs.append({
                "idea_id": idea["idea_id"],
                "run": run + 1,
                "raw_response": response,
                "parsed_scores": scores
            })

            print(f"Run {run+1} for {idea['idea_id']} done.")

    # Save results
    pd.DataFrame(results).to_csv(csv_file, index=False)
    with open(json_file, "w") as f:
        json.dump(logs, f, indent=2)
    return results, logs


# Execute scoring
run_benchmark(IDEAS, bootstrap_runs=10)