import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from app.agents import OrchestratorAgent
from app.logger import AgentLogger


TEST_CASES = [
    {
        "name": "empty_input",
        "text": "",
        "expect_error": True,
    },
    {
        "name": "whitespace_only",
        "text": "   ",
        "expect_error": True,
    },
    {
        "name": "very_short_input",
        "text": "Invoice",
        "expect_error": False,
    },
    {
        "name": "mixed_content",
        "text": "John Smith submitted an invoice for consulting work in Q1 2024 with a total of 2500 MAD.",
        "expect_error": False,
    },
    {
        "name": "invoice_without_amount",
        "text": "Invoice #9001 for design services. Client: Nova. Contact: billing@nova.com",
        "expect_error": False,
    },
    {
        "name": "cv_without_name",
        "text": "Experienced Python developer with 4 years experience in machine learning and Docker.",
        "expect_error": False,
    },
    {
        "name": "report_without_metrics",
        "text": "Quarterly business report discussing internal process improvements and planning priorities.",
        "expect_error": False,
    },
    {
        "name": "off_domain_text",
        "text": "The weather is pleasant today and the city center is crowded.",
        "expect_error": False,
    },
]


def main():
    logs_dir = os.path.join(project_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    logger = AgentLogger(os.path.join(logs_dir, "edge_case_agent_logs.json"))
    orchestrator = OrchestratorAgent(logger, os.path.join(project_dir, "model"))
    orchestrator.initialize()

    results = []

    for case in TEST_CASES:
        record = {
            "name": case["name"],
            "input": case["text"],
            "expect_error": case["expect_error"],
        }

        try:
            result = orchestrator.process_document(
                case["text"],
                require_human_approval=False,
            )
            record["status"] = "success"
            record["classification"] = result["classification"]["category"]
            record["confidence"] = result["classification"]["confidence"]
            record["summary"] = result["summary"]["summary"]
            record["passed"] = not case["expect_error"]
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            record["passed"] = case["expect_error"]

        results.append(record)

    output_path = os.path.join(logs_dir, "edge_case_results.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    total_passed = sum(1 for item in results if item["passed"])
    print(f"Edge-case checks complete: {total_passed}/{len(results)} passed")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
