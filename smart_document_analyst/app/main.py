"""
Smart Document Analyst - Main CLI Application
=============================================
Simple CLI application for document analysis
Multi-agent system with:
- OrchestratorAgent: Coordinates workflow
- ClassifierAgent: Classifies documents using PyTorch
- ExtractionAgent: Extracts key information
- SummaryAgent: Generates summaries
"""

import os
import sys
import json
from datetime import datetime

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

# Add project directory to path for imports
sys.path.insert(0, project_dir)

from app.agents import OrchestratorAgent
from app.logger import AgentLogger
from app.tools import DocumentClassifierTool


def print_header():
    """Print application header"""
    print("\n" + "=" * 60)
    print("       SMART DOCUMENT ANALYST")
    print("       Multi-Agent AI System")
    print("=" * 60)
    print("\n📄 Document Classification System")
    print("   - Classifies: CV, Invoice, Report")
    print("   - Extracts key information")
    print("   - Generates summaries")
    print("   - Human-in-the-loop approval")
    print("=" * 60 + "\n")


def print_menu():
    """Print main menu"""
    print("\n" + "-" * 40)
    print("MAIN MENU")
    print("-" * 40)
    print("1. Analyze a document")
    print("2. Train the model")
    print("3. View logs")
    print("4. Generate final report")
    print("5. Exit")
    print("-" * 40)


def analyze_document(orchestrator):
    """
    Analyze a document through the multi-agent system
    
    IMPROVEMENT 3: Enhanced error handling for empty input, missing files, and invalid output
    
    Args:
        orchestrator: OrchestratorAgent instance
    """
    print("\n" + "=" * 60)
    print("DOCUMENT ANALYSIS")
    print("=" * 60)
    
    # Get document text from user
    print("\nEnter the document text (paste or type):")
    print("-" * 40)
    
    lines = []
    print("(Press Enter twice to finish input)")
    
    while True:
        try:
            line = input()
            if line == "" and (not lines or lines[-1] == ""):
                break
            lines.append(line)
        except EOFError:
            break
    
    document_text = "\n".join(lines).strip()
    
    # Error handling for empty input
    if not document_text or not document_text.strip():
        print("\n❌ Error: Empty document text provided!")
        print("   Please enter some text to analyze.")
        return None
    
    # Process the document
    try:
        print("\n🔄 Processing document...")
        result = orchestrator.process_document(document_text, require_human_approval=True)
        
        if result['status'] == 'success':
            print("\n" + "=" * 60)
            print("✅ ANALYSIS COMPLETE")
            print("=" * 60)
            
            # Display results
            print(f"\n📋 Classification:")
            print(f"   Category: {result['classification']['category']}")
            print(f"   Confidence: {result['classification']['confidence']:.2%}")
            
            print(f"\n📄 Extraction:")
            for key, value in result['extraction']['extracted_fields'].items():
                print(f"   {key}: {value}")
            
            # Also show common extracted fields for invoices
            extraction = result['extraction']
            if extraction.get('amounts'):
                print(f"   amounts: {', '.join(extraction['amounts'])}")
            if extraction.get('dates'):
                print(f"   dates: {', '.join(extraction['dates'])}")
            if extraction.get('emails'):
                print(f"   emails: {', '.join(extraction['emails'])}")
            
            print(f"\n📝 Summary:")
            print(f"   {result['summary']['summary']}")
            
            if result.get('human_approval'):
                print(f"\n✅ Human Approval: {'Approved' if result['human_approval']['approved'] else 'Rejected'}")
            
            return result
        
        elif result['status'] == 'rejected':
            print("\n❌ Analysis rejected by user.")
            return result
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        return None


def train_model():
    """
    Train the PyTorch model
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    # Get the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, "model")
    
    # Check if model already exists
    model_path = os.path.join(model_dir, "document_classifier.pt")
    if os.path.exists(model_path):
        response = input("\n⚠️  Model already exists. Retrain? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Training cancelled.")
            return
    
    # Run training
    print("\n🔄 Starting model training...")
    print("   This may take a few minutes...")
    
    try:
        # Import and run training
        from model.train_model import main as train_main
        train_main()
        print("\n✅ Model training complete!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("   Make sure you have all required dependencies installed.")


def view_logs(logger):
    """
    View agent logs
    
    Args:
        logger: AgentLogger instance
    """
    print("\n" + "=" * 60)
    print("AGENT LOGS")
    print("=" * 60)
    
    logs = logger.get_logs(limit=20)
    
    if not logs:
        print("\nNo logs available.")
        return
    
    print(f"\nShowing {len(logs)} most recent log entries:\n")
    
    for i, log in enumerate(logs, 1):
        timestamp = log.get('timestamp', 'N/A')
        agent = log.get('agent', 'Unknown')
        action = log.get('action', 'No action')
        details = log.get('details', {})
        
        print(f"{i}. [{timestamp}]")
        print(f"   Agent: {agent}")
        print(f"   Action: {action}")
        if details:
            print(f"   Details: {details}")
        print()


def generate_final_report(orchestrator, logger):
    """
    Generate a final report from the last analysis
    
    Args:
        orchestrator: OrchestratorAgent instance
        logger: AgentLogger instance
    """
    print("\n" + "=" * 60)
    print("FINAL REPORT GENERATION")
    print("=" * 60)
    
    # Get workflow state
    state = orchestrator.get_state()
    
    if not state or 'classification' not in state:
        print("\n❌ No analysis data available.")
        print("   Please analyze a document first.")
        return
    
    # Get the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Generate report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("SMART DOCUMENT ANALYST - FINAL REPORT")
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Classification
    if 'classification' in state:
        report_lines.append("CLASSIFICATION RESULTS")
        report_lines.append("-" * 40)
        report_lines.append(f"Category: {state['classification']['category']}")
        report_lines.append(f"Confidence: {state['classification']['confidence']:.2%}")
        report_lines.append("")
    
    # Extraction
    if 'extraction' in state:
        report_lines.append("EXTRACTED INFORMATION")
        report_lines.append("-" * 40)
        for key, value in state['extraction'].get('extracted_fields', {}).items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")
    
    # Summary
    if 'summary' in state:
        report_lines.append("DOCUMENT SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(state['summary'].get('summary', 'N/A'))
        report_lines.append("")
    
    # Human approval
    if 'human_approval' in state:
        report_lines.append("HUMAN APPROVAL")
        report_lines.append("-" * 40)
        approval = state['human_approval']
        report_lines.append(f"Status: {'Approved' if approval.get('approved') else 'Rejected'}")
        report_lines.append(f"Timestamp: {approval.get('timestamp', 'N/A')}")
        report_lines.append("")
    
    # Agent logs summary
    report_lines.append("AGENT ACTIVITY LOG")
    report_lines.append("-" * 40)
    summary = logger.get_log_summary()
    report_lines.append(f"Total log entries: {summary['total_logs']}")
    for agent, count in summary['agents'].items():
        report_lines.append(f"  {agent}: {count} actions")
    report_lines.append("")
    
    report_lines.append("=" * 60)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 60)
    
    # Write to file
    report_text = "\n".join(report_lines)
    
    # Get the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    report_file = os.path.join(project_dir, "final_report.txt")
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✅ Final report generated: {report_file}")
        print("\nReport Preview:")
        print("-" * 40)
        print(report_text)
        
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")


def check_model_exists():
    """
    Check if the model files exist
    
    Returns:
        bool: True if model exists, False otherwise
    """
    # Get the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, "model")
    
    model_path = os.path.join(model_dir, "document_classifier.pt")
    vectorizer_path = os.path.join(model_dir, "document_classifier_vectorizer.pkl")
    label_encoder_path = os.path.join(model_dir, "document_classifier_label_encoder.pkl")
    
    return (os.path.exists(model_path) and 
            os.path.exists(vectorizer_path) and 
            os.path.exists(label_encoder_path))


def main():
    """
    Main CLI application
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Change to project directory
    os.chdir(project_dir)
    
    # Print header
    print_header()
    
    # Initialize logger
    logger = AgentLogger(os.path.join(project_dir, "logs", "agent_logs.json"))
    logger.log({
        "agent": "System",
        "action": "application_started",
        "details": {"version": "1.0"}
    })
    
    # Check if model exists
    if not check_model_exists():
        print("⚠️  Model not found. Please train the model first.")
        print("   Select option 2 from the menu to train.")
    
    # Initialize orchestrator
    try:
        orchestrator = OrchestratorAgent(
            logger,
            os.path.join(project_dir, "model"),
            ollama_model="phi3:latest"
        )   
        orchestrator.initialize()
    except FileNotFoundError as e:
        print(f"\n⚠️  Warning: {e}")
        print("   The model needs to be trained first.")
        orchestrator = None
    except Exception as e:
        print(f"\n❌ Error initializing agents: {e}")
        orchestrator = None
    
    # Main loop
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
        except EOFError:
            choice = '5'
        
        if choice == '1':
            # Analyze document
            if orchestrator is None:
                print("\n❌ Cannot analyze: Model not initialized.")
                print("   Please train the model first (option 2).")
            else:
                analyze_document(orchestrator)
        
        elif choice == '2':
            # Train model
            train_model()
            
            # Reinitialize orchestrator after training
            if check_model_exists():
                try:
                    orchestrator = OrchestratorAgent(
                        logger,
                        os.path.join(project_dir, "model"),
                        ollama_model="phi3:latest"
                    )
                    orchestrator.initialize()
                    print("\n✅ Agents reinitialized with new model.")
                except Exception as e:
                    print(f"\n⚠️  Warning: Could not reinitialize agents: {e}")
        
        elif choice == '3':
            # View logs
            view_logs(logger)
        
        elif choice == '4':
            # Generate final report
            if orchestrator:
                generate_final_report(orchestrator, logger)
            else:
                print("\n❌ No analysis data available.")
        
        elif choice == '5':
            # Exit
            print("\n👋 Thank you for using Smart Document Analyst!")
            logger.log({
                "agent": "System",
                "action": "application_exited",
                "details": {}
            })
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()
