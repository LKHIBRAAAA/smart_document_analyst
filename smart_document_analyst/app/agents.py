"""
Smart Document Analyst - Agents Module
=======================================
Defines the multi-agent system:
1. OrchestratorAgent - Coordinates the workflow
2. ClassifierAgent - Classifies documents using PyTorch model
3. ExtractionAgent - Extracts key information
4. SummaryAgent - Generates summaries
"""

import json
import os
from datetime import datetime
from app.tools import DocumentClassifierTool, ExtractionTool, SummaryTool, OllamaSummaryTool

class BaseAgent:
    """
    Base class for all agents with common functionality
    """
    
    def __init__(self, name, logger):
        self.name = name
        self.logger = logger
        self.tool = None
    
    def log_action(self, action, details=None):
        """Log agent action with timestamp"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "action": action,
            "details": details or {}
        }
        self.logger.log(log_entry)
        print(f"[{self.name}] {action}")
        if details:
            for key, value in details.items():
                print(f"  - {key}: {value}")
    
    def execute(self, *args, **kwargs):
        """Execute the agent's main task - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement execute()")


class ClassifierAgent(BaseAgent):
    """
    Classifier Agent - Uses PyTorch model to classify documents
    """
    
    def __init__(self, logger, model_dir="model"):
        super().__init__("ClassifierAgent", logger)
        self.model_dir = model_dir
        self.tool = None
    
    def initialize(self):
        """Initialize the classifier tool"""
        try:
            self.tool = DocumentClassifierTool(self.model_dir)
            self.log_action("initialized", {"status": "success", "model_dir": self.model_dir})
            return True
        except FileNotFoundError as e:
            self.log_action("initialization_failed", {"error": str(e)})
            raise
        except Exception as e:
            self.log_action("initialization_failed", {"error": str(e)})
            raise
    
    def execute(self, document_text):
        """
        Classify the document text
        
        Args:
            document_text: The text to classify
            
        Returns:
            dict: Classification result
        """
        if not document_text or not document_text.strip():
            raise ValueError("Empty document text provided")
        
        self.log_action("classifying_document", {"text_length": len(document_text)})
        
        try:
            result = self.tool.classify(document_text)
            self.log_action("classification_complete", {
                "category": result['category'],
                "confidence": f"{result['confidence']:.2%}"
            })
            return result
        except Exception as e:
            self.log_action("classification_failed", {"error": str(e)})
            raise


class ExtractionAgent(BaseAgent):
    """
    Extraction Agent - Extracts key information from documents
    """
    
    def __init__(self, logger):
        super().__init__("ExtractionAgent", logger)
        self.tool = ExtractionTool()
    
    def initialize(self):
        """Initialize the extraction tool"""
        self.log_action("initialized", {"status": "success"})
        return True
    
    def execute(self, document_text, category):
        """
        Extract key information from the document
        
        Args:
            document_text: The document text
            category: The document category
            
        Returns:
            dict: Extracted information
        """
        if not document_text or not document_text.strip():
            raise ValueError("Empty document text provided")
        
        if not category:
            raise ValueError("Category is required for extraction")
        
        self.log_action("extracting_information", {
            "category": category,
            "text_length": len(document_text)
        })
        
        try:
            result = self.tool.extract(document_text, category)
            self.log_action("extraction_complete", {
                "fields_extracted": len(result.get('extracted_fields', {}))
            })
            return result
        except Exception as e:
            self.log_action("extraction_failed", {"error": str(e)})
            raise


class SummaryAgent(BaseAgent):
    """
    Summary Agent - Generates summaries of documents
    """
    
    def __init__(self, logger, model_name="phi3:latest"):
        super().__init__("SummaryAgent", logger)
        self.model_name = model_name
        self.tool = SummaryTool()
        self.ollama_tool = None
    
    def initialize(self):
        """Initialize the summary tools"""
        try:
            self.ollama_tool = OllamaSummaryTool(self.model_name)
            self.log_action("initialized", {
                "status": "success",
                "ollama_enabled": True,
                "ollama_model": self.model_name
            })
        except Exception:
            self.ollama_tool = None
            self.log_action("initialized", {
                "status": "success",
                "ollama_enabled": False
            })
        return True
    
    def execute(self, document_text, category, extracted_info):
        """
        Generate a summary of the document
        
        Args:
            document_text: The document text
            category: The document category
            extracted_info: Extracted key information
            
        Returns:
            dict: Summary information
        """
        if not document_text or not document_text.strip():
            raise ValueError("Empty document text provided")
        
        if not category:
            raise ValueError("Category is required for summarization")
        
        self.log_action("generating_summary", {
            "category": category,
            "text_length": len(document_text)
        })
        
        try:
            if self.ollama_tool is not None:
                try:
                    result = self.ollama_tool.summarize(document_text, category, extracted_info)
                    self.log_action("summary_generated_with_ollama", {
                        "summary_length": len(result.get("summary", "")),
                        "ollama_model": self.model_name
                    })
                    return result
                except Exception as ollama_error:
                    self.log_action("ollama_summary_failed", {"error": str(ollama_error)})

            result = self.tool.summarize(document_text, category, extracted_info)
            self.log_action("summary_complete", {
                "summary_length": len(result.get('summary', '')),
                "fallback": "rule_based"
            })
            return result
        except Exception as e:
            self.log_action("summarization_failed", {"error": str(e)})
            raise


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent - Coordinates the workflow between specialist agents
    Includes human-in-the-loop checkpoint
    """
    
    def __init__(self, logger, model_dir="model", ollama_model="phi3:latest"):
        super().__init__("OrchestratorAgent", logger)
        self.model_dir = model_dir
        self.ollama_model = ollama_model
        self.classifier_agent = None
        self.extraction_agent = None
        self.summary_agent = None
        self.workflow_state = {}
    
    def initialize(self):
        """Initialize all specialist agents"""
        self.log_action("initializing_agents", {})

        # Initialize classifier agent
        self.classifier_agent = ClassifierAgent(self.logger, self.model_dir)
        self.classifier_agent.initialize()

        # Initialize extraction agent
        self.extraction_agent = ExtractionAgent(self.logger)
        self.extraction_agent.initialize()

        # Initialize summary agent
        self.summary_agent = SummaryAgent(self.logger, self.ollama_model)
        self.summary_agent.initialize()

        self.log_action("all_agents_initialized", {"status": "ready"})
        return True
    
    def process_document(self, document_text, require_human_approval=True):
        """
        Process a document through the complete workflow
        
        Args:
            document_text: The document text to process
            require_human_approval: Whether to require human approval before final report
            
        Returns:
            dict: Complete processing result
        """
        if not document_text or not document_text.strip():
            raise ValueError("Empty document text provided")
        
        self.log_action("starting_document_processing", {"text_length": len(document_text)})
        
        # Step 1: Classify the document
        self.log_action("step_1_classification", {"agent": "ClassifierAgent"})
        classification_result = self.classifier_agent.execute(document_text)
        category = classification_result['category']
        self.workflow_state['classification'] = classification_result
        
        # Step 2: Extract key information
        self.log_action("step_2_extraction", {"agent": "ExtractionAgent"})
        extraction_result = self.extraction_agent.execute(document_text, category)
        self.workflow_state['extraction'] = extraction_result
        
        # Step 3: Generate summary
        self.log_action("step_3_summarization", {"agent": "SummaryAgent"})
        summary_result = self.summary_agent.execute(document_text, category, extraction_result)
        self.workflow_state['summary'] = summary_result
        
        # Step 4: Human-in-the-loop checkpoint
        approval_status = None
        if require_human_approval:
            self.log_action("human_approval_required", {"status": "pending"})
            approval_status = self._request_human_approval(classification_result, extraction_result, summary_result)
            self.workflow_state['human_approval'] = approval_status
            
            if not approval_status['approved']:
                self.log_action("human_rejected", {"reason": approval_status.get('reason', 'Unknown')})
                return {
                    "status": "rejected",
                    "message": "Document processing rejected by user",
                    "details": self.workflow_state
                }
        
        # Step 5: Generate final result
        self.log_action("workflow_complete", {"status": "success"})
        
        return {
            "status": "success",
            "classification": classification_result,
            "extraction": extraction_result,
            "summary": summary_result,
            "human_approval": approval_status
        }
    
    def _request_human_approval(self, classification, extraction, summary):
        """
        Request human approval for the processing result
        
        Args:
            classification: Classification result
            extraction: Extraction result
            summary: Summary result
            
        Returns:
            dict: Approval status
        """
        print("\n" + "=" * 60)
        print("HUMAN-IN-THE-LOOP CHECKPOINT")
        print("=" * 60)
        
        # Display classification
        print(f"\n📋 Document Classification:")
        print(f"   Category: {classification['category']}")
        print(f"   Confidence: {classification['confidence']:.2%}")
        
        # Display extracted info
        print(f"\n📄 Extracted Information:")
        for key, value in extraction.get('extracted_fields', {}).items():
            print(f"   {key}: {value}")
        
        # Display summary
        print(f"\n📝 Summary:")
        print(f"   {summary['summary']}")
        
        # Request approval
        print("\n" + "-" * 60)
        response = input("Do you approve this analysis? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y', '1']:
            return {"approved": True, "timestamp": datetime.now().isoformat()}
        else:
            return {"approved": False, "reason": "User rejected", "timestamp": datetime.now().isoformat()}
    
    def get_state(self):
        """Get current workflow state"""
        return self.workflow_state
    
    def reset_state(self):
        """Reset workflow state"""
        self.workflow_state = {}
        self.log_action("state_reset", {})


def create_agents(logger):
    """
    Factory function to create all agents
    
    Args:
        logger: Logger instance
        
    Returns:
        dict: Dictionary of agents
    """
    # Create orchestrator
    orchestrator = OrchestratorAgent(logger)
    orchestrator.initialize()
    
    return {
        "orchestrator": orchestrator,
        "classifier": orchestrator.classifier_agent,
        "extraction": orchestrator.extraction_agent,
        "summary": orchestrator.summary_agent
    }
