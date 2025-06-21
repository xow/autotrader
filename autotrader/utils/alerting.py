"""
Alerting system for the AutoTrader application.

This module provides functionality for creating and sending alerts
when errors or important events occur in the system.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ErrorAlert:
    """
    Represents an error alert in the system.
    
    Attributes:
        error_type: The type/category of the error.
        timestamp: ISO format timestamp when the alert was created.
        message: Human-readable description of the alert.
        severity: Severity level (e.g., 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
        context: Additional context data as a dictionary.
    """
    error_type: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    message: str = ""
    severity: str = "MEDIUM"
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the alert to a dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize the alert to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorAlert':
        """Create an ErrorAlert from a dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ErrorAlert':
        """Create an ErrorAlert from a JSON string."""
        return cls.from_dict(json.loads(json_str))


class AlertingSystem:
    """
    Centralized alerting system for the AutoTrader application.
    
    Handles the creation, routing, and delivery of alerts based on severity
    and configuration. Supports multiple notification channels (email, etc.).
    """
    
    def __init__(self, email_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the alerting system.
        
        Args:
            email_config: Configuration for email alerts. Should include:
                         - smtp_server: SMTP server hostname
                         - smtp_port: SMTP server port
                         - username: SMTP username
                         - password: SMTP password
                         - recipient: Email recipient address
                         - sender: Optional sender email (defaults to username)
        """
        self.email_config = email_config or {}
        self.alert_counts: Dict[str, int] = {}
        self.alert_timestamps: Dict[str, List[float]] = {}
        self.alert_thresholds = {
            'LOW': 10,     # Max 10 low-severity alerts per hour
            'MEDIUM': 5,   # Max 5 medium-severity alerts per hour
            'HIGH': 3,     # Max 3 high-severity alerts per hour
            'CRITICAL': 0  # Always alert for critical issues
        }
    
    def should_alert(self, error_type: str, severity: str = 'MEDIUM') -> bool:
        """
        Determine if an alert should be sent for the given error type and severity.
        
        Implements rate limiting based on error type and severity. For non-critical
        alerts, this method will only return True if the number of recent alerts
        meets or exceeds the threshold for the given severity level.
        
        Args:
            error_type: The type/category of the error.
            severity: The severity level of the alert.
            
        Returns:
            bool: True if an alert should be sent, False otherwise.
        """
        severity = severity.upper()
        
        # Always allow critical alerts
        if severity == 'CRITICAL':
            return True
            
        # Check if we've exceeded the rate limit for this error type
        now = datetime.utcnow().timestamp()
        one_hour_ago = now - 3600  # 1 hour in seconds
        
        # Clean up old timestamps
        if error_type in self.alert_timestamps:
            self.alert_timestamps[error_type] = [
                ts for ts in self.alert_timestamps[error_type] if ts > one_hour_ago
            ]
        
        # Get the threshold for this severity
        threshold = self.alert_thresholds.get(severity, 5)  # Default to 5 if severity not found
        
        # Initialize if needed
        if error_type not in self.alert_timestamps:
            self.alert_timestamps[error_type] = []
        
        # For non-critical alerts, only alert if we've reached the threshold
        # This means the first (threshold) errors won't trigger an alert
        current_count = len(self.alert_timestamps[error_type])
        
        # Increment the count for this error type
        self.alert_timestamps[error_type].append(now)
        
        # Only return True if we've reached or exceeded the threshold
        # This means the first (threshold) errors won't trigger an alert
        return current_count >= threshold
    
    def create_alert(
        self, 
        error_type: str, 
        message: str = "", 
        severity: str = "MEDIUM", 
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorAlert:
        """
        Create a new alert.
        
        Args:
            error_type: The type/category of the error.
            message: Human-readable description of the alert.
            severity: Severity level (e.g., 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
            context: Additional context data as a dictionary.
            
        Returns:
            ErrorAlert: The created alert.
        """
        return ErrorAlert(
            error_type=error_type,
            message=message,
            severity=severity.upper(),
            context=context or {}
        )
    
    def send_alert(self, alert: ErrorAlert) -> bool:
        """
        Send an alert through the appropriate channels.
        
        Args:
            alert: The alert to send.
            
        Returns:
            bool: True if the alert was sent successfully, False otherwise.
        """
        try:
            # Check if we should send this alert based on rate limiting
            if not self.should_alert(alert.error_type, alert.severity):
                logger.debug(
                    f"Rate limiting prevented alert for {alert.error_type} "
                    f"with severity {alert.severity}"
                )
                return False
            
            # Send email if configured
            if self.email_config:
                self._send_email_alert(alert)
            
            # TODO: Add other notification channels (e.g., Slack, PagerDuty)
            
            logger.info(f"Sent alert: {alert.error_type} - {alert.message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}", exc_info=True)
            return False
    
    def _send_email_alert(self, alert: ErrorAlert) -> None:
        """
        Send an alert via email.
        
        Args:
            alert: The alert to send.
            
        Raises:
            ValueError: If email configuration is incomplete.
            smtplib.SMTPException: If sending the email fails.
        """
        required_fields = ['smtp_server', 'smtp_port', 'username', 'password', 'recipient']
        missing = [field for field in required_fields if field not in self.email_config]
        if missing:
            raise ValueError(f"Missing required email configuration: {', '.join(missing)}")
        
        # Create message
        sender = self.email_config.get('sender', self.email_config['username'])
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = self.email_config['recipient']
        msg['Subject'] = f"[{alert.severity}] {alert.error_type}"
        
        # Format message body
        body = f"""
        Alert Type: {alert.error_type}
        Severity: {alert.severity}
        Time: {alert.timestamp}
        
        Message:
        {alert.message}
        
        Context:
        {json.dumps(alert.context, indent=2, default=str)}
        """
        
        msg.attach(MIMEText(body.strip(), 'plain'))
        
        # Connect to SMTP server and send
        try:
            # Try with context manager first (Python 3.3+)
            with smtplib.SMTP(
                self.email_config['smtp_server'], 
                self.email_config['smtp_port']
            ) as server:
                server.starttls()
                server.login(
                    self.email_config['username'], 
                    self.email_config['password']
                )
                server.send_message(msg)
        except (AttributeError, TypeError):
            # Fall back to manual connection handling if context manager not supported
            server = smtplib.SMTP(
                self.email_config['smtp_server'], 
                self.email_config['smtp_port']
            )
            try:
                server.starttls()
                server.login(
                    self.email_config['username'], 
                    self.email_config['password']
                )
                server.send_message(msg)
            finally:
                try:
                    server.quit()
                except:
                    pass


# Default instance for convenience
alerting_system = AlertingSystem()
