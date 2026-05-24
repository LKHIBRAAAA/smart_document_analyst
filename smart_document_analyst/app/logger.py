"""
Smart Document Analyst - Logger Module
=======================================
Provides JSON logging with timestamps for all agent actions
"""

import json
import os
from datetime import datetime
from pathlib import Path


class AgentLogger:
    """
    Logger class for logging agent actions in JSON format with timestamps
    """
    
    def __init__(self, log_file="logs/agent_logs.json"):
        """
        Initialize the logger
        
        Args:
            log_file: Path to the log file
        """
        self.log_file = log_file
        self.logs = []
        
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Load existing logs if file exists
        self._load_logs()
    
    def _load_logs(self):
        """Load existing logs from file"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.logs = json.load(f)
            except json.JSONDecodeError:
                # If file is corrupted, start fresh
                self.logs = []
            except Exception as e:
                print(f"Warning: Could not load existing logs: {e}")
                self.logs = []
    
    def _save_logs(self):
        """Save logs to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            print(f"Error saving logs: {e}")
    
    def log(self, entry):
        """
        Add a log entry with timestamp
        
        Args:
            entry: Dictionary containing log information
        """
        # Ensure timestamp is present
        if 'timestamp' not in entry:
            entry['timestamp'] = datetime.now().isoformat()
        
        # Add to logs list
        self.logs.append(entry)
        
        # Save to file
        self._save_logs()
        
        # Also print to console
        self._print_log(entry)
    
    def _print_log(self, entry):
        """Print log entry to console"""
        timestamp = entry.get('timestamp', '')
        agent = entry.get('agent', 'Unknown')
        action = entry.get('action', 'No action')
        
        print(f"[{timestamp}] {agent}: {action}")
    
    def get_logs(self, agent=None, limit=None):
        """
        Get logs, optionally filtered by agent
        
        Args:
            agent: Optional agent name to filter by
            limit: Maximum number of logs to return
            
        Returns:
            list: Log entries
        """
        logs = self.logs
        
        if agent:
            logs = [log for log in logs if log.get('agent') == agent]
        
        if limit:
            logs = logs[-limit:]
        
        return logs
    
    def get_recent_logs(self, count=10):
        """
        Get the most recent log entries
        
        Args:
            count: Number of recent logs to return
            
        Returns:
            list: Recent log entries
        """
        return self.logs[-count:] if self.logs else []
    
    def clear_logs(self):
        """Clear all logs"""
        self.logs = []
        self._save_logs()
    
    def get_log_summary(self):
        """
        Get a summary of all logs
        
        Returns:
            dict: Summary statistics
        """
        summary = {
            "total_logs": len(self.logs),
            "agents": {},
            "actions": {}
        }
        
        for log in self.logs:
            # Count by agent
            agent = log.get('agent', 'Unknown')
            summary['agents'][agent] = summary['agents'].get(agent, 0) + 1
            
            # Count by action
            action = log.get('action', 'Unknown')
            summary['actions'][action] = summary['actions'].get(action, 0) + 1
        
        return summary
    
    def export_logs(self, output_file):
        """
        Export logs to a specific file
        
        Args:
            output_file: Path to output file
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
            print(f"Logs exported to: {output_file}")
        except Exception as e:
            print(f"Error exporting logs: {e}")


def create_logger(log_file="logs/agent_logs.json"):
    """
    Factory function to create a logger
    
    Args:
        log_file: Path to the log file
        
    Returns:
        AgentLogger: Logger instance
    """
    return AgentLogger(log_file)


# Test the logger when run directly
if __name__ == "__main__":
    print("Testing Agent Logger...")
    
    # Create a test logger
    logger = create_logger("logs/test_logs.json")
    
    # Log some test entries
    logger.log({
        "agent": "TestAgent",
        "action": "test_action",
        "details": {"key": "value"}
    })
    
    logger.log({
        "agent": "AnotherAgent",
        "action": "another_action",
        "details": {"data": "test"}
    })
    
    # Get summary
    print("\nLog Summary:")
    print(json.dumps(logger.get_log_summary(), indent=2))
    
    # Get recent logs
    print("\nRecent Logs:")
    print(json.dumps(logger.get_recent_logs(5), indent=2))