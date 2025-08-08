"""
Alerting and Notification System for MeshAI

This module provides comprehensive alerting capabilities including
multiple notification channels, alert routing, and escalation policies.
"""

import asyncio
import aiohttp
import smtplib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import json
import threading

import structlog
from .performance_monitor import Alert, AlertSeverity

logger = structlog.get_logger(__name__)


class NotificationChannel(str, Enum):
    """Available notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    DISCORD = "discord"
    TEAMS = "teams"


class EscalationLevel(str, Enum):
    """Alert escalation levels"""
    LEVEL_1 = "level_1"  # Initial alert
    LEVEL_2 = "level_2"  # Escalate after timeout
    LEVEL_3 = "level_3"  # Executive escalation
    RESOLVED = "resolved"


@dataclass
class NotificationConfig:
    """Configuration for a notification channel"""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    retry_attempts: int = 3
    retry_delay: timedelta = timedelta(minutes=1)
    rate_limit: Optional[timedelta] = None  # Min time between notifications


@dataclass
class AlertRoute:
    """Alert routing rule"""
    name: str
    conditions: Dict[str, Any]  # Conditions to match
    channels: List[str]  # Notification channel names
    escalation_timeout: timedelta = timedelta(minutes=15)
    enabled: bool = True
    severity_filter: Optional[List[AlertSeverity]] = None
    agent_filter: Optional[List[str]] = None
    business_hours_only: bool = False


@dataclass
class EscalationPolicy:
    """Alert escalation policy"""
    name: str
    levels: List[Dict[str, Any]]  # Each level has timeout and channels
    repeat_interval: Optional[timedelta] = None
    max_escalations: int = 3


@dataclass
class NotificationAttempt:
    """Record of a notification attempt"""
    alert_id: str
    channel: str
    timestamp: datetime
    success: bool
    response_time: float
    error: Optional[str] = None
    escalation_level: EscalationLevel = EscalationLevel.LEVEL_1


@dataclass
class AlertNotification:
    """Alert notification being processed"""
    alert: Alert
    route: AlertRoute
    escalation_level: EscalationLevel = EscalationLevel.LEVEL_1
    attempts: List[NotificationAttempt] = field(default_factory=list)
    last_escalation: Optional[datetime] = None
    acknowledged: bool = False
    resolved: bool = False


class NotificationSender:
    """Base class for notification senders"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def send(self, alert: Alert, message: str, **kwargs) -> bool:
        """Send notification - to be implemented by subclasses"""
        raise NotImplementedError


class EmailNotificationSender(NotificationSender):
    """Email notification sender"""
    
    async def send(self, alert: Alert, message: str, **kwargs) -> bool:
        try:
            smtp_server = self.config.get('smtp_server', 'localhost')
            smtp_port = self.config.get('smtp_port', 587)
            username = self.config.get('username')
            password = self.config.get('password')
            from_email = self.config.get('from_email')
            to_emails = self.config.get('to_emails', [])
            
            if not to_emails:
                logger.warning("No email recipients configured")
                return False
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] MeshAI Alert: {alert.rule_name}"
            
            # Create HTML body
            html_body = f"""
            <html>
            <body>
                <h2>MeshAI Alert</h2>
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>Agent:</strong> {alert.agent_id}</p>
                <p><strong>Metric:</strong> {alert.metric_name}</p>
                <p><strong>Current Value:</strong> {alert.current_value}</p>
                <p><strong>Threshold:</strong> {alert.threshold}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                <p><strong>Triggered At:</strong> {alert.triggered_at}</p>
                <hr>
                <p>{message}</p>
            </body>
            </html>
            """
            
            msg.attach(MimeText(html_body, 'html'))
            
            # Send email
            await asyncio.get_event_loop().run_in_executor(
                None, self._send_email, smtp_server, smtp_port, username, password, msg
            )
            
            logger.info(f"Email alert sent for {alert.rule_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_email(self, smtp_server, smtp_port, username, password, msg):
        """Send email synchronously"""
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            if username and password:
                server.starttls()
                server.login(username, password)
            server.send_message(msg)


class SlackNotificationSender(NotificationSender):
    """Slack notification sender"""
    
    async def send(self, alert: Alert, message: str, **kwargs) -> bool:
        try:
            webhook_url = self.config.get('webhook_url')
            channel = self.config.get('channel', '#alerts')
            username = self.config.get('username', 'MeshAI Alerts')
            
            if not webhook_url:
                logger.warning("No Slack webhook URL configured")
                return False
            
            # Create Slack message
            color_map = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning", 
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger"
            }
            
            payload = {
                "channel": channel,
                "username": username,
                "attachments": [{
                    "color": color_map.get(alert.severity, "danger"),
                    "title": f"MeshAI Alert: {alert.rule_name}",
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Agent", "value": alert.agent_id, "short": True},
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Value", "value": str(alert.current_value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True},
                        {"title": "Time", "value": alert.triggered_at.strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                    ],
                    "text": message,
                    "footer": "MeshAI Monitoring",
                    "ts": int(alert.triggered_at.timestamp())
                }]
            }
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    success = response.status == 200
                    if success:
                        logger.info(f"Slack alert sent for {alert.rule_name}")
                    else:
                        logger.error(f"Slack alert failed: {response.status}")
                    return success
                    
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class WebhookNotificationSender(NotificationSender):
    """Generic webhook notification sender"""
    
    async def send(self, alert: Alert, message: str, **kwargs) -> bool:
        try:
            webhook_url = self.config.get('url')
            headers = self.config.get('headers', {})
            method = self.config.get('method', 'POST').upper()
            
            if not webhook_url:
                logger.warning("No webhook URL configured")
                return False
            
            # Create payload
            payload = {
                "alert": {
                    "rule_name": alert.rule_name,
                    "agent_id": alert.agent_id,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "triggered_at": alert.triggered_at.isoformat()
                },
                "notification_message": message,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                if method == 'POST':
                    async with session.post(webhook_url, json=payload, headers=headers) as response:
                        success = 200 <= response.status < 300
                elif method == 'PUT':
                    async with session.put(webhook_url, json=payload, headers=headers) as response:
                        success = 200 <= response.status < 300
                else:
                    logger.error(f"Unsupported webhook method: {method}")
                    return False
                
                if success:
                    logger.info(f"Webhook alert sent for {alert.rule_name}")
                else:
                    logger.error(f"Webhook alert failed: {response.status}")
                return success
                    
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class PagerDutyNotificationSender(NotificationSender):
    """PagerDuty notification sender"""
    
    async def send(self, alert: Alert, message: str, **kwargs) -> bool:
        try:
            integration_key = self.config.get('integration_key')
            
            if not integration_key:
                logger.warning("No PagerDuty integration key configured")
                return False
            
            # Create PagerDuty event
            payload = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "dedup_key": f"meshai_alert_{alert.rule_name}_{alert.agent_id}",
                "payload": {
                    "summary": f"MeshAI Alert: {alert.rule_name}",
                    "source": alert.agent_id,
                    "severity": "critical" if alert.severity == AlertSeverity.CRITICAL else "error",
                    "component": "meshai",
                    "group": "agents",
                    "class": alert.metric_name,
                    "custom_details": {
                        "metric_name": alert.metric_name,
                        "current_value": alert.current_value,
                        "threshold": alert.threshold,
                        "message": message
                    }
                }
            }
            
            # Send to PagerDuty
            pagerduty_url = "https://events.pagerduty.com/v2/enqueue"
            async with aiohttp.ClientSession() as session:
                async with session.post(pagerduty_url, json=payload) as response:
                    success = response.status == 202
                    if success:
                        logger.info(f"PagerDuty alert sent for {alert.rule_name}")
                    else:
                        logger.error(f"PagerDuty alert failed: {response.status}")
                    return success
                    
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False


class AlertingSystem:
    """
    Comprehensive alerting and notification system.
    
    Features:
    - Multiple notification channels
    - Alert routing and filtering
    - Escalation policies
    - Rate limiting and deduplication
    - Notification delivery tracking
    """
    
    def __init__(self):
        # Configuration
        self.notification_configs: Dict[str, NotificationConfig] = {}
        self.alert_routes: Dict[str, AlertRoute] = {}
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        
        # State management
        self.active_notifications: Dict[str, AlertNotification] = {}
        self.notification_history: List[NotificationAttempt] = []
        self.last_notifications: Dict[str, datetime] = {}  # Rate limiting
        
        # Senders
        self.senders: Dict[NotificationChannel, NotificationSender] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("Alerting system initialized")
    
    def configure_email(
        self,
        name: str,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: List[str],
        enabled: bool = True
    ):
        """Configure email notifications"""
        config = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            enabled=enabled,
            config={
                'smtp_server': smtp_server,
                'smtp_port': smtp_port,
                'username': username,
                'password': password,
                'from_email': from_email,
                'to_emails': to_emails
            }
        )
        
        self.notification_configs[name] = config
        self.senders[NotificationChannel.EMAIL] = EmailNotificationSender(config.config)
        
        logger.info(f"Configured email notifications: {name}")
    
    def configure_slack(
        self,
        name: str,
        webhook_url: str,
        channel: str = '#alerts',
        username: str = 'MeshAI Alerts',
        enabled: bool = True
    ):
        """Configure Slack notifications"""
        config = NotificationConfig(
            channel=NotificationChannel.SLACK,
            enabled=enabled,
            config={
                'webhook_url': webhook_url,
                'channel': channel,
                'username': username
            }
        )
        
        self.notification_configs[name] = config
        self.senders[NotificationChannel.SLACK] = SlackNotificationSender(config.config)
        
        logger.info(f"Configured Slack notifications: {name}")
    
    def configure_webhook(
        self,
        name: str,
        url: str,
        method: str = 'POST',
        headers: Optional[Dict[str, str]] = None,
        enabled: bool = True
    ):
        """Configure webhook notifications"""
        config = NotificationConfig(
            channel=NotificationChannel.WEBHOOK,
            enabled=enabled,
            config={
                'url': url,
                'method': method,
                'headers': headers or {}
            }
        )
        
        self.notification_configs[name] = config
        self.senders[NotificationChannel.WEBHOOK] = WebhookNotificationSender(config.config)
        
        logger.info(f"Configured webhook notifications: {name}")
    
    def configure_pagerduty(
        self,
        name: str,
        integration_key: str,
        enabled: bool = True
    ):
        """Configure PagerDuty notifications"""
        config = NotificationConfig(
            channel=NotificationChannel.PAGERDUTY,
            enabled=enabled,
            config={'integration_key': integration_key}
        )
        
        self.notification_configs[name] = config
        self.senders[NotificationChannel.PAGERDUTY] = PagerDutyNotificationSender(config.config)
        
        logger.info(f"Configured PagerDuty notifications: {name}")
    
    def add_alert_route(self, route: AlertRoute):
        """Add an alert routing rule"""
        with self._lock:
            self.alert_routes[route.name] = route
            logger.info(f"Added alert route: {route.name}")
    
    def add_escalation_policy(self, policy: EscalationPolicy):
        """Add an escalation policy"""
        with self._lock:
            self.escalation_policies[policy.name] = policy
            logger.info(f"Added escalation policy: {policy.name}")
    
    async def start_processing(self):
        """Start alert processing"""
        if self._running:
            logger.warning("Alerting system already running")
            return
        
        self._running = True
        self._processing_task = asyncio.create_task(self._processing_loop())
        logger.info("Alerting system started")
    
    async def stop_processing(self):
        """Stop alert processing"""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Alerting system stopped")
    
    async def process_alert(self, alert: Alert):
        """Process an incoming alert"""
        try:
            # Find matching routes
            matching_routes = self._find_matching_routes(alert)
            
            if not matching_routes:
                logger.debug(f"No matching routes for alert: {alert.rule_name}")
                return
            
            # Create notifications for each matching route
            for route in matching_routes:
                await self._create_notification(alert, route)
                
        except Exception as e:
            logger.error(f"Error processing alert {alert.rule_name}: {e}")
    
    def _find_matching_routes(self, alert: Alert) -> List[AlertRoute]:
        """Find alert routes that match the given alert"""
        matching_routes = []
        
        with self._lock:
            for route in self.alert_routes.values():
                if not route.enabled:
                    continue
                
                # Check severity filter
                if (route.severity_filter and 
                    alert.severity not in route.severity_filter):
                    continue
                
                # Check agent filter
                if (route.agent_filter and 
                    alert.agent_id not in route.agent_filter):
                    continue
                
                # Check business hours (simplified)
                if route.business_hours_only:
                    current_hour = datetime.now().hour
                    if not (9 <= current_hour <= 17):  # 9 AM to 5 PM
                        continue
                
                # Check custom conditions
                if self._matches_conditions(alert, route.conditions):
                    matching_routes.append(route)
        
        return matching_routes
    
    def _matches_conditions(self, alert: Alert, conditions: Dict[str, Any]) -> bool:
        """Check if alert matches route conditions"""
        for key, expected_value in conditions.items():
            if key == 'metric_name':
                if alert.metric_name != expected_value:
                    return False
            elif key == 'agent_id_pattern':
                import re
                if not re.match(expected_value, alert.agent_id):
                    return False
            # Add more condition types as needed
        
        return True
    
    async def _create_notification(self, alert: Alert, route: AlertRoute):
        """Create a notification for an alert and route"""
        notification_id = f"{alert.rule_name}_{alert.agent_id}_{route.name}"
        
        with self._lock:
            # Check if we already have an active notification
            if notification_id in self.active_notifications:
                logger.debug(f"Notification already active: {notification_id}")
                return
            
            # Create new notification
            notification = AlertNotification(
                alert=alert,
                route=route
            )
            
            self.active_notifications[notification_id] = notification
        
        # Send initial notifications
        await self._send_notifications(notification_id)
    
    async def _send_notifications(self, notification_id: str):
        """Send notifications for a specific alert"""
        with self._lock:
            if notification_id not in self.active_notifications:
                return
            
            notification = self.active_notifications[notification_id]
        
        # Send to all configured channels for this route
        for channel_name in notification.route.channels:
            if channel_name in self.notification_configs:
                config = self.notification_configs[channel_name]
                if config.enabled:
                    await self._send_single_notification(notification, channel_name, config)
    
    async def _send_single_notification(
        self,
        notification: AlertNotification,
        channel_name: str,
        config: NotificationConfig
    ):
        """Send a single notification"""
        # Check rate limiting
        rate_limit_key = f"{channel_name}_{notification.alert.rule_name}"
        if self._is_rate_limited(rate_limit_key, config.rate_limit):
            logger.debug(f"Rate limited notification: {rate_limit_key}")
            return
        
        # Create message
        message = self._create_message(notification.alert, notification.escalation_level)
        
        # Find appropriate sender
        sender = self.senders.get(config.channel)
        if not sender:
            logger.error(f"No sender configured for channel: {config.channel}")
            return
        
        # Send with retries
        for attempt in range(config.retry_attempts):
            start_time = asyncio.get_event_loop().time()
            
            try:
                success = await sender.send(notification.alert, message)
                response_time = asyncio.get_event_loop().time() - start_time
                
                # Record attempt
                attempt_record = NotificationAttempt(
                    alert_id=f"{notification.alert.rule_name}_{notification.alert.agent_id}",
                    channel=channel_name,
                    timestamp=datetime.utcnow(),
                    success=success,
                    response_time=response_time,
                    escalation_level=notification.escalation_level
                )
                
                with self._lock:
                    notification.attempts.append(attempt_record)
                    self.notification_history.append(attempt_record)
                
                if success:
                    # Update rate limiting
                    self.last_notifications[rate_limit_key] = datetime.utcnow()
                    logger.info(f"Notification sent successfully: {channel_name}")
                    break
                else:
                    logger.warning(f"Notification failed (attempt {attempt + 1}): {channel_name}")
                    
            except Exception as e:
                response_time = asyncio.get_event_loop().time() - start_time
                
                # Record failed attempt
                attempt_record = NotificationAttempt(
                    alert_id=f"{notification.alert.rule_name}_{notification.alert.agent_id}",
                    channel=channel_name,
                    timestamp=datetime.utcnow(),
                    success=False,
                    response_time=response_time,
                    error=str(e),
                    escalation_level=notification.escalation_level
                )
                
                with self._lock:
                    notification.attempts.append(attempt_record)
                    self.notification_history.append(attempt_record)
                
                logger.error(f"Notification error (attempt {attempt + 1}): {e}")
            
            # Wait before retry
            if attempt < config.retry_attempts - 1:
                await asyncio.sleep(config.retry_delay.total_seconds())
    
    def _is_rate_limited(self, key: str, rate_limit: Optional[timedelta]) -> bool:
        """Check if notification is rate limited"""
        if not rate_limit:
            return False
        
        last_notification = self.last_notifications.get(key)
        if not last_notification:
            return False
        
        return datetime.utcnow() - last_notification < rate_limit
    
    def _create_message(self, alert: Alert, escalation_level: EscalationLevel) -> str:
        """Create notification message"""
        escalation_prefix = ""
        if escalation_level != EscalationLevel.LEVEL_1:
            escalation_prefix = f"[{escalation_level.value.upper()}] "
        
        message = f"""
{escalation_prefix}Alert: {alert.rule_name}

Agent: {alert.agent_id}
Metric: {alert.metric_name}
Current Value: {alert.current_value}
Threshold: {alert.threshold}
Severity: {alert.severity.value.upper()}
Triggered: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

Message: {alert.message}
        """.strip()
        
        return message
    
    async def _processing_loop(self):
        """Main processing loop for escalations and cleanup"""
        while self._running:
            try:
                await self._process_escalations()
                await self._cleanup_resolved_alerts()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alerting processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_escalations(self):
        """Process alert escalations"""
        current_time = datetime.utcnow()
        
        with self._lock:
            notifications_to_escalate = []
            
            for notification_id, notification in self.active_notifications.items():
                if notification.resolved or notification.acknowledged:
                    continue
                
                # Check if escalation timeout has passed
                if (notification.last_escalation and
                    current_time - notification.last_escalation >= notification.route.escalation_timeout):
                    notifications_to_escalate.append(notification_id)
                elif (not notification.last_escalation and
                      current_time - notification.alert.triggered_at >= notification.route.escalation_timeout):
                    notifications_to_escalate.append(notification_id)
        
        # Process escalations
        for notification_id in notifications_to_escalate:
            await self._escalate_notification(notification_id)
    
    async def _escalate_notification(self, notification_id: str):
        """Escalate a notification to the next level"""
        with self._lock:
            if notification_id not in self.active_notifications:
                return
            
            notification = self.active_notifications[notification_id]
            
            # Update escalation level
            if notification.escalation_level == EscalationLevel.LEVEL_1:
                notification.escalation_level = EscalationLevel.LEVEL_2
            elif notification.escalation_level == EscalationLevel.LEVEL_2:
                notification.escalation_level = EscalationLevel.LEVEL_3
            else:
                return  # Already at max escalation
            
            notification.last_escalation = datetime.utcnow()
        
        logger.warning(f"Escalating alert: {notification_id} to {notification.escalation_level.value}")
        
        # Send escalated notifications
        await self._send_notifications(notification_id)
    
    async def _cleanup_resolved_alerts(self):
        """Clean up resolved alerts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        with self._lock:
            # Remove old resolved notifications
            to_remove = [
                notification_id for notification_id, notification in self.active_notifications.items()
                if notification.resolved and notification.alert.resolved_at and 
                   notification.alert.resolved_at < cutoff_time
            ]
            
            for notification_id in to_remove:
                del self.active_notifications[notification_id]
            
            # Clean up old notification history
            self.notification_history = [
                attempt for attempt in self.notification_history
                if attempt.timestamp > cutoff_time
            ]
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert to stop escalations"""
        with self._lock:
            for notification in self.active_notifications.values():
                if f"{notification.alert.rule_name}_{notification.alert.agent_id}" == alert_id:
                    notification.acknowledged = True
                    logger.info(f"Alert acknowledged: {alert_id}")
                    break
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        with self._lock:
            for notification in self.active_notifications.values():
                if f"{notification.alert.rule_name}_{notification.alert.agent_id}" == alert_id:
                    notification.resolved = True
                    logger.info(f"Alert resolved: {alert_id}")
                    break
    
    def get_active_notifications(self) -> List[Dict[str, Any]]:
        """Get all active notifications"""
        with self._lock:
            return [
                {
                    "id": f"{n.alert.rule_name}_{n.alert.agent_id}",
                    "rule_name": n.alert.rule_name,
                    "agent_id": n.alert.agent_id,
                    "severity": n.alert.severity.value,
                    "escalation_level": n.escalation_level.value,
                    "acknowledged": n.acknowledged,
                    "resolved": n.resolved,
                    "attempts": len(n.attempts),
                    "triggered_at": n.alert.triggered_at.isoformat()
                }
                for n in self.active_notifications.values()
            ]
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        with self._lock:
            total_attempts = len(self.notification_history)
            successful_attempts = sum(1 for a in self.notification_history if a.success)
            
            # Channel statistics
            channel_stats = {}
            for attempt in self.notification_history:
                if attempt.channel not in channel_stats:
                    channel_stats[attempt.channel] = {"total": 0, "successful": 0}
                channel_stats[attempt.channel]["total"] += 1
                if attempt.success:
                    channel_stats[attempt.channel]["successful"] += 1
            
            return {
                "total_attempts": total_attempts,
                "successful_attempts": successful_attempts,
                "success_rate": successful_attempts / total_attempts if total_attempts > 0 else 0,
                "active_notifications": len(self.active_notifications),
                "channel_stats": channel_stats
            }


# Global alerting system instance
alerting_system = AlertingSystem()