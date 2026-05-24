"""
Smart Document Analyst - Tools Module
=====================================
Loads the trained PyTorch model and provides classification functionality
"""

import torch
import torch.nn as nn
import pickle
import os
import json
import urllib.request
import urllib.error
from datetime import datetime


class DocumentClassifierNN(nn.Module):
    """
    Neural Network for Document Classification (same architecture as training)
    """
    def __init__(self, input_size, num_classes):
        super(DocumentClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class DocumentClassifierTool:
    """
    Tool for classifying documents using the trained PyTorch model
    """
    
    def __init__(self, model_dir="model"):
        """
        Initialize the classifier tool by loading model artifacts
        
        Args:
            model_dir: Directory containing model files
        """
        import os
        self.model_dir = os.path.abspath(model_dir)
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_loaded = False
        
        # Load model artifacts
        self._load_artifacts()
    
    def _load_artifacts(self):
        """
        Load the trained model, vectorizer, and label encoder
        """
        model_path = os.path.join(self.model_dir, "document_classifier.pt")
        vectorizer_path = os.path.join(self.model_dir, "document_classifier_vectorizer.pkl")
        label_encoder_path = os.path.join(self.model_dir, "document_classifier_label_encoder.pkl")
        
        # Check if all files exist
        missing_files = []
        if not os.path.exists(model_path):
            missing_files.append(model_path)
        if not os.path.exists(vectorizer_path):
            missing_files.append(vectorizer_path)
        if not os.path.exists(label_encoder_path):
            missing_files.append(label_encoder_path)
        
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {missing_files}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        input_size = checkpoint['input_size']
        num_classes = checkpoint['num_classes']
        
        self.model = DocumentClassifierNN(input_size, num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.is_loaded = True
        print(f"Model loaded successfully on {self.device}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
    
    def classify(self, text):
        """
        Classify a document text into one of the categories
        
        Args:
            text: The document text to classify
            
        Returns:
            dict: Classification result with category, confidence, and probabilities
        """
        # Error handling: check for empty input
        if not text or not text.strip():
            raise ValueError("Empty text provided for classification")
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Please train the model first.")
        
        # Transform text using TF-IDF vectorizer
        text_tfidf = self.vectorizer.transform([text])
        
        # Convert to PyTorch tensor
        text_tensor = torch.FloatTensor(text_tfidf.toarray()).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(text_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Get results
        predicted_label = self.label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]
        confidence = probabilities[0][predicted_class.item()].item()
        all_probs = {
            label: prob.item() 
            for label, prob in zip(self.label_encoder.classes_, probabilities[0])
        }
        
        # IMPROVEMENT 2: Add confidence warning if < 60%
        # This helps identify uncertain predictions
        if confidence < 0.60:
            print("\n[WARNING] Low confidence prediction!")
            print(f"   Confidence: {confidence:.2%} (threshold: 60%)")
        
        return {
            "category": predicted_label,
            "confidence": confidence,
            "probabilities": all_probs
        }
    
    def batch_classify(self, texts):
        """
        Classify multiple documents at once
        
        Args:
            texts: List of document texts
            
        Returns:
            list: List of classification results
        """
        results = []
        for text in texts:
            try:
                result = self.classify(text)
                results.append(result)
            except Exception as e:
                results.append({
                    "category": "unknown",
                    "confidence": 0.0,
                    "error": str(e)
                })
        return results


class ExtractionTool:
    """
    Tool for extracting key information from documents based on category
    Enhanced with regex patterns for amounts, dates, emails, and keywords
    """
    
    def __init__(self):
        """Initialize the extraction tool"""
        import re
        # Regex patterns for extracting common fields
        self.amount_patterns = [
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(MAD|DH|EUR|USD|CAD)',  # With currency
            r'(?:Total|Amount|Sum):\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',  # With label
        ]
        self.date_pattern = r'(\d{2}/\d{2}/\d{4})'  # DD/MM/YYYY format
        self.email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    
    def extract(self, text, category):
        """
        Extract key information from document based on its category
        
        Args:
            text: The document text
            category: The document category (cv, invoice, report)
            
        Returns:
            dict: Extracted information with structured fields
        """
        # Error handling: return safe empty structure if text is empty
        if not text or not text.strip():
            return {
                "category": category,
                "extracted_fields": {},
                "dates": [],
                "amounts": [],
                "emails": [],
                "keywords": []
            }
        
        extracted = {
            "category": category,
            "extracted_fields": {},
            "dates": [],
            "amounts": [],
            "emails": [],
            "keywords": []
        }
        
        # Extract common fields using regex
        extracted = self._extract_common_fields(text, extracted)
        
        # Extract based on category
        if category.lower() == "cv":
            extracted["extracted_fields"] = self._extract_cv_info(text)
        elif category.lower() == "invoice":
            extracted["extracted_fields"] = self._extract_invoice_info(text)
        elif category.lower() == "report":
            extracted["extracted_fields"] = self._extract_report_info(text)
        else:
            extracted["extracted_fields"] = {"note": "Unknown category"}
        
        return extracted
    
    def _extract_common_fields(self, text, extracted):
        """
        Extract common fields (dates, amounts, emails, keywords) using regex
        
        Args:
            text: Document text
            extracted: Dictionary to add extracted fields to
            
        Returns:
            dict: Updated extracted dictionary
        """
        import re
        
        # Extract amounts (MAD, DH, USD, EUR, CAD)
        amount_pattern = r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(MAD|DH|EUR|USD|CAD|\$)'
        amounts = re.findall(amount_pattern, text, re.IGNORECASE)
        extracted["amounts"] = [f"{amt} {curr}" for amt, curr in amounts[:5]]  # Limit to 5
        
        # Extract dates (DD/MM/YYYY format)
        date_pattern = r'(\d{2}/\d{2}/\d{4})'
        dates = re.findall(date_pattern, text)
        extracted["dates"] = dates[:5]  # Limit to 5
        
        # Extract emails
        email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        emails = re.findall(email_pattern, text)
        extracted["emails"] = emails[:5]  # Limit to 5
        
        # Extract keywords (first 10 meaningful words)
        words = text.split()
        # Filter out short words and punctuation
        keywords = [w.strip('.,!?;:()[]{}') for w in words if len(w) > 3][:10]
        extracted["keywords"] = keywords
        
        return extracted
    
    def _extract_cv_info(self, text):
        """Extract information from CV/resume documents"""
        info = {}
        text_lower = text.lower()
        
        # Extract a likely person name while avoiding common sentence starters.
        import re
        stop_pairs = {
            "Experienced Python",
            "Senior Python",
            "Software Engineer",
            "Project Manager",
            "Data Analyst",
            "Machine Learning",
            "Frontend Developer",
            "Backend Developer",
        }
        names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)
        for name in names:
            if name not in stop_pairs:
                info["candidate_name"] = name
                break
        
        # Extract skills
        skill_keywords = ["python", "java", "javascript", "sql", "aws", "docker", "kubernetes", 
                         "machine learning", "deep learning", "nlp", "data science"]
        found_skills = [s for s in skill_keywords if s in text_lower]
        if found_skills:
            info["skills"] = found_skills
        
        # Extract experience years
        exp_match = re.search(r'(\d+)\s+years?\s+experience', text_lower)
        if exp_match:
            info["years_experience"] = exp_match.group(1)
        
        # Extract education
        education_keywords = ["phd", "master", "bachelor", "mba", "bsc", "msc"]
        for edu in education_keywords:
            if edu in text_lower:
                info["education"] = edu.upper()
                break
        
        return info
    
    def _extract_invoice_info(self, text):
        """
        Extract information from invoice documents
        Enhanced to extract date DD/MM/YYYY, email, and amount with currency
        """
        import re
        info = {}
        text_lower = text.lower()
        
        # Extract invoice number
        inv_match = re.search(r'invoice\s*#?(\d+)', text_lower)
        if inv_match:
            info["invoice_number"] = inv_match.group(1)
        
        # Extract amount with currency (MAD, DH, USD, EUR, CAD)
        amount_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(MAD|DH|EUR|USD|CAD|\$)', text, re.IGNORECASE)
        if amount_match:
            info["amount"] = f"{amount_match.group(1)} {amount_match.group(2).upper()}"
        else:
            # Fallback only for labeled totals to avoid confusing invoice numbers with amounts.
            amount_match = re.search(
                r'(?:total|amount|sum|payment due|total amount)\s*:?\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
                text,
                re.IGNORECASE,
            )
            if amount_match:
                info["amount"] = f"${amount_match.group(1)}"
        
        # Extract date (DD/MM/YYYY format)
        date_match = re.search(r'(\d{2}/\d{2}/\d{4})', text)
        if date_match:
            info["date"] = date_match.group(1)
        else:
            # Try YYYY-MM-DD format and convert
            date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
            if date_match:
                # Convert to DD/MM/YYYY
                year, month, day = date_match.groups()
                info["date"] = f"{day}/{month}/{year}"
        
        # Extract email
        email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text)
        if email_match:
            info["email"] = email_match.group(1)
        
        # Extract client
        client_match = re.search(r'client:?\s*([A-Z][a-zA-Z]+)', text)
        if client_match:
            info["client"] = client_match.group(1)
        
        return info
    
    def _extract_report_info(self, text):
        """Extract information from report documents"""
        info = {}
        text_lower = text.lower()
        
        # Extract quarter/year
        import re
        quarter_match = re.search(r'q(\d)\s+(\d{4})', text_lower)
        if quarter_match:
            info["quarter"] = f"Q{quarter_match.group(1)}"
            info["year"] = quarter_match.group(2)
        
        # Extract revenue
        revenue_match = re.search(r'revenue:?\s*\$?(\d+(?:\.\d+)?)\s*[Mm]?', text)
        if revenue_match:
            info["revenue"] = f"${revenue_match.group(1)}M"
        
        # Extract profit
        profit_match = re.search(r'net profit:?\s*\$?(\d+(?:\.\d+)?)\s*[Mm]?', text)
        if profit_match:
            info["net_profit"] = f"${profit_match.group(1)}K"
        
        # Extract growth percentage
        growth_match = re.search(r'(\d+)%\s*(?:growth|increase|up)', text_lower)
        if growth_match:
            info["growth"] = f"{growth_match.group(1)}%"
        
        return info

class OllamaSummaryTool:
    """
    Tool for generating summaries using a local Ollama model.
    Falls back to the rule-based summary tool if Ollama is unavailable.
    """

    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def summarize(self, text, category, extracted_info):
        if not text or not text.strip():
            raise ValueError("Empty text provided for summarization")

        prompt = self._build_prompt(text, category, extracted_info)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        summary_text = data.get("response", "").strip()
        if not summary_text:
            raise RuntimeError("Ollama returned an empty summary")

        return {
            "category": category,
            "summary": summary_text,
            "word_count": len(text.split()),
            "key_points": []
        }

    def _build_prompt(self, text, category, extracted_info):
        return f"""You are a document summarization assistant.

Document category: {category}

Extracted information:
{json.dumps(extracted_info, indent=2)}

Document text:
{text}

Write a short, professional summary in 1 to 3 sentences.
If information is missing, do not invent facts.
Use only the provided document text and extracted information.
"""
class SummaryTool:
    """
    Tool for generating document summaries
    """
    
    def summarize(self, text, category, extracted_info):
        """
        Generate a summary of the document
        
        Args:
            text: The original document text
            category: The document category
            extracted_info: Extracted key information
            
        Returns:
            dict: Summary information
        """
        if not text or not text.strip():
            raise ValueError("Empty text provided for summarization")
        
        # Generate summary based on category
        summary_text = self._generate_summary(text, category, extracted_info)
        
        return {
            "category": category,
            "summary": summary_text,
            "word_count": len(text.split()),
            "key_points": self._extract_key_points(text, category)
        }
    
    def _generate_summary(self, text, category, extracted_info):
        """
        Generate a human-readable summary
        Updated to use common fields (dates, amounts, emails) for invoices
        """
        fields = extracted_info.get("extracted_fields", {})
        
        # Get common fields too
        dates = extracted_info.get("dates", [])
        amounts = extracted_info.get("amounts", [])
        emails = extracted_info.get("emails", [])
        
        if category.lower() == "cv":
            name = fields.get("candidate_name")
            skills = fields.get("skills", [])
            exp = fields.get("years_experience", "N/A")
            subject = name if name else "an unnamed candidate"
            return f"CV for {subject} with {exp} years of experience. Key skills: {', '.join(skills[:3]) if skills else 'Not specified'}."
        
        elif category.lower() == "invoice":
            inv_num = fields.get("invoice_number", "Unknown")
            # Use amount with currency from extracted_fields or fallback to common amounts
            amount = fields.get("amount")
            if not amount and amounts:
                amount = amounts[0]
            if not amount:
                amount = "N/A"
            
            # Use date from extracted_fields or fallback to common dates
            date = fields.get("date")
            if not date and dates:
                date = dates[0]
            if not date:
                date = "N/A"
            
            # Use email from extracted_fields or fallback to common emails
            email = fields.get("email")
            if not email and emails:
                email = emails[0]
            
            # Build summary with available info
            summary = f"Invoice #{inv_num}"
            if amount != "N/A":
                summary += f" for {amount}"
            else:
                summary += " with amount not specified"
            if date != "N/A":
                summary += f", dated {date}"
            if email:
                summary += f" (email: {email})"
            summary += "."
            return summary
        
        elif category.lower() == "report":
            quarter = fields.get("quarter")
            year = fields.get("year")
            revenue = fields.get("revenue")
            growth = fields.get("growth")
            net_profit = fields.get("net_profit")

            parts = []
            if quarter and year:
                parts.append(f"{quarter} {year} report")
            elif quarter:
                parts.append(f"{quarter} report")
            elif year:
                parts.append(f"Report for {year}")
            else:
                parts.append("Report document")

            details = []
            if revenue:
                details.append(f"revenue of {revenue}")
            if net_profit:
                details.append(f"net profit of {net_profit}")
            if growth:
                details.append(f"growth of {growth}")

            if details:
                return parts[0] + " showing " + ", ".join(details) + "."
            return parts[0] + " with limited structured metrics extracted."
        
        return "Document summary generated."
    
    def _extract_key_points(self, text, category):
        """Extract key points from the document"""
        points = []
        
        # Simple key point extraction
        sentences = text.split('.')
        for sentence in sentences[:3]:  # Take first 3 sentences
            sentence = sentence.strip()
            if len(sentence) > 20:
                points.append(sentence[:100] + "..." if len(sentence) > 100 else sentence)
        
        return points


def get_available_tools():
    """
    Get dictionary of available tools
    """
    return {
        "classifier": DocumentClassifierTool,
        "extractor": ExtractionTool,
        "summarizer": SummaryTool
    }


# Test the tools when run directly
if __name__ == "__main__":
    print("Testing Document Classifier Tool...")
    
    try:
        # Try to load the classifier
        classifier = DocumentClassifierTool("model")
        
        # Test with sample texts
        test_texts = [
            "Experienced software developer with 5 years in Python and JavaScript.",
            "Invoice #12345 for web development services. Amount: $5000.",
            "Quarterly financial report Q4 2023. Revenue: $2.5M, up 15% YoY."
        ]
        
        for text in test_texts:
            result = classifier.classify(text)
            print(f"\nText: {text[:50]}...")
            print(f"Category: {result['category']}")
            print(f"Confidence: {result['confidence']:.2%}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_model.py first to create the model.")
    except Exception as e:
        print(f"Error: {e}")
