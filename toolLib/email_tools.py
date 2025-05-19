import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, List
import logging
from ..utils.logging import get_logger

class EmailTool(ToolBase):
    """Tool for email operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the email tool
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        super().__init__(config)
        self.logger = get_logger(self.__class__.__name__)
        self.smtp_server = config['smtp_server']
        self.smtp_port = config['smtp_port']
        self.smtp_user = config['smtp_user']
        self.smtp_password = config['smtp_password']
    
    def send_email(self, 
                  to: str, 
                  subject: str, 
                  body: str, 
                  attachments: List[str] = None) -> Dict[str, Any]:
        """
        Send an email
        
        Args:
            to (str): Recipient email address
            subject (str): Email subject
            body (str): Email body
            attachments (List[str]): List of file paths to attach
            
        Returns:
            Dict[str, Any]: Operation result
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = to
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachments if any
            if attachments:
                for attachment in attachments:
                    try:
                        with open(attachment, 'rb') as file:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(file.read())
                            encoders.encode_base64(part)
                            part.add_header('Content-Disposition', 
                                           f'attachment; filename={os.path.basename(attachment)}')
                            msg.attach(part)
                    except Exception as e:
                        self.logger.warning(f"Failed to attach file {attachment}: {str(e)}")
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            return {
                'success': True,
                'to': to,
                'subject': subject,
                'attachments': attachments or []
            }
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def validate(self, task: Dict[str, Any]) -> bool:
        """
        Validate an email task
        
        Args:
            task (Dict[str, Any]): Task to validate
            
        Returns:
            bool: True if task is valid, False otherwise
        """
        required_fields = ['to', 'subject', 'body']
        return all(field in task for field in required_fields)
