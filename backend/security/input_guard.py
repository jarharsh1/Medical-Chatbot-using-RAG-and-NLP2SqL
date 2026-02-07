"""
Input Guard - Security middleware for query validation.

This module provides:
1. Prompt injection detection - Blocks LLM manipulation attempts
2. SQL injection pattern detection - Extra layer beyond SQL_BANNED_OPS
3. Medical safety warnings - Flags dangerous medical queries
4. Rate limiting support - Optional request throttling

Integration:
- Called by app.py before processing queries
- Returns SecurityCheckResult with threat level and details
- Logs security events for monitoring

Design Philosophy:
- Lightweight (~200 lines vs 800+ in deleted branch)
- Actually integrated into request pipeline
- Focused on real threats, not theoretical ones
"""

import re
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    NONE = "none"           # Safe input
    LOW = "low"             # Minor concern, allow with logging
    MEDIUM = "medium"       # Suspicious, allow with warning
    HIGH = "high"           # Likely attack, block
    CRITICAL = "critical"   # Definite attack, block and log


@dataclass
class SecurityCheckResult:
    """Result of security check on user input."""
    is_safe: bool
    threat_level: ThreatLevel
    threats_detected: List[str] = field(default_factory=list)
    sanitized_input: Optional[str] = None
    warning_message: Optional[str] = None
    should_block: bool = False

    def to_dict(self) -> Dict:
        return {
            "is_safe": self.is_safe,
            "threat_level": self.threat_level.value,
            "threats_detected": self.threats_detected,
            "warning_message": self.warning_message,
            "should_block": self.should_block,
        }


class InputGuard:
    """
    Security guard for validating and sanitizing user inputs.

    Usage:
        guard = InputGuard()
        result = guard.check(user_input)
        if result.should_block:
            return error_response(result.warning_message)
    """

    # Prompt injection patterns (case-insensitive)
    PROMPT_INJECTION_PATTERNS = [
        # Direct instruction override
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(previous|prior|above)",
        r"forget\s+(all\s+)?(previous|prior|above|everything)",
        r"override\s+(all\s+)?(previous|prior|system)",

        # Role manipulation
        r"you\s+are\s+now\s+(DAN|evil|unrestricted|jailbroken)",
        r"pretend\s+(you\s+are|to\s+be)\s+(a\s+)?(different|evil|unrestricted)",
        r"act\s+as\s+(if\s+)?(you\s+have\s+)?no\s+(restrictions?|limits?|rules?)",
        r"roleplay\s+as\s+(an?\s+)?(unrestricted|evil|different)",

        # System prompt extraction
        r"(show|reveal|display|output|print)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?)",
        r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?)",
        r"repeat\s+(back\s+)?(your|the)\s+(system\s+)?(prompt|instructions?)",

        # Delimiter injection
        r"\[SYSTEM\]",
        r"\[INST\]",
        r"<\|system\|>",
        r"<\|user\|>",
        r"<\|assistant\|>",
        r"```system",

        # Output manipulation
        r"(respond|reply|answer)\s+(only\s+)?(with|using)\s+(json|xml|code)",
        r"output\s+(only|just)\s+(the\s+)?(raw|all)\s+(data|patient|sql)",
    ]

    # SQL injection patterns (beyond basic keyword blocking)
    SQL_INJECTION_PATTERNS = [
        r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)",
        r"--\s*$",                          # SQL comment at end
        r"/\*.*\*/",                         # Block comments
        r"UNION\s+(ALL\s+)?SELECT",          # Union injection
        r"OR\s+1\s*=\s*1",                   # Always true
        r"AND\s+1\s*=\s*0",                  # Always false
        r"'\s*OR\s+'",                       # String OR injection
        r"EXEC(\s+|\()",                     # Execute
        r"xp_cmdshell",                      # SQL Server command
        r"INFORMATION_SCHEMA",               # Schema enumeration
        r"SLEEP\s*\(",                       # Time-based injection
        r"BENCHMARK\s*\(",                   # Time-based injection
        r"WAITFOR\s+DELAY",                  # SQL Server delay
    ]

    # Medical safety patterns (queries that need warnings)
    MEDICAL_SAFETY_PATTERNS = [
        (r"(maximum|lethal|fatal|deadly|overdose)\s+(safe\s+)?(dose|dosage|amount)",
         "Questions about maximum or lethal doses require professional medical guidance."),
        (r"(how\s+to|can\s+I)\s+(kill|harm|hurt|poison)",
         "This system cannot provide information that could cause harm."),
        (r"suicide|self[- ]?harm|end\s+my\s+life",
         "If you're in crisis, please contact emergency services or a crisis helpline."),
        (r"(buy|purchase|obtain)\s+(drugs?|medications?|pills?)\s+(without|no)\s+(prescription|rx)",
         "This system cannot assist with obtaining medications without proper prescriptions."),
    ]

    def __init__(self, enable_rate_limiting: bool = False, rate_limit_per_minute: int = 30):
        self.enable_rate_limiting = enable_rate_limiting
        self.rate_limit_per_minute = rate_limit_per_minute
        self._request_counts: Dict[str, List[float]] = defaultdict(list)

        # Compile patterns for efficiency
        self._prompt_injection_re = [
            re.compile(p, re.IGNORECASE) for p in self.PROMPT_INJECTION_PATTERNS
        ]
        self._sql_injection_re = [
            re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS
        ]
        self._medical_safety_re = [
            (re.compile(p, re.IGNORECASE), msg) for p, msg in self.MEDICAL_SAFETY_PATTERNS
        ]

    def check(self, user_input: str, client_id: Optional[str] = None) -> SecurityCheckResult:
        """
        Check user input for security threats.

        Args:
            user_input: The raw user query
            client_id: Optional client identifier for rate limiting

        Returns:
            SecurityCheckResult with threat assessment
        """
        if not user_input or not user_input.strip():
            return SecurityCheckResult(
                is_safe=False,
                threat_level=ThreatLevel.LOW,
                threats_detected=["empty_input"],
                warning_message="Please enter a question.",
                should_block=True,
            )

        threats: List[str] = []
        threat_level = ThreatLevel.NONE
        warning_message = None

        # 1. Check for prompt injection
        prompt_threats = self._check_prompt_injection(user_input)
        if prompt_threats:
            threats.extend(prompt_threats)
            threat_level = ThreatLevel.HIGH
            warning_message = "Your query contains patterns that could manipulate the AI system."
            logger.warning(f"Prompt injection detected: {user_input[:100]}...")

        # 2. Check for SQL injection
        sql_threats = self._check_sql_injection(user_input)
        if sql_threats:
            threats.extend(sql_threats)
            if threat_level.value < ThreatLevel.HIGH.value:
                threat_level = ThreatLevel.HIGH
            warning_message = "Your query contains potentially dangerous SQL patterns."
            logger.warning(f"SQL injection pattern detected: {user_input[:100]}...")

        # 3. Check for medical safety concerns
        medical_warning = self._check_medical_safety(user_input)
        if medical_warning:
            threats.append("medical_safety_concern")
            if threat_level == ThreatLevel.NONE:
                threat_level = ThreatLevel.MEDIUM
            warning_message = medical_warning
            logger.info(f"Medical safety flag: {user_input[:100]}...")

        # 4. Rate limiting (optional)
        if self.enable_rate_limiting and client_id:
            if self._is_rate_limited(client_id):
                threats.append("rate_limited")
                threat_level = ThreatLevel.MEDIUM
                warning_message = "Too many requests. Please wait a moment."
                return SecurityCheckResult(
                    is_safe=False,
                    threat_level=threat_level,
                    threats_detected=threats,
                    warning_message=warning_message,
                    should_block=True,
                )

        # Determine if we should block
        should_block = threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)

        return SecurityCheckResult(
            is_safe=len(threats) == 0,
            threat_level=threat_level,
            threats_detected=threats,
            sanitized_input=user_input.strip(),
            warning_message=warning_message,
            should_block=should_block,
        )

    def _check_prompt_injection(self, text: str) -> List[str]:
        """Check for prompt injection patterns."""
        detected = []
        for pattern in self._prompt_injection_re:
            if pattern.search(text):
                detected.append(f"prompt_injection:{pattern.pattern[:30]}")
        return detected

    def _check_sql_injection(self, text: str) -> List[str]:
        """Check for SQL injection patterns."""
        detected = []
        for pattern in self._sql_injection_re:
            if pattern.search(text):
                detected.append(f"sql_injection:{pattern.pattern[:30]}")
        return detected

    def _check_medical_safety(self, text: str) -> Optional[str]:
        """Check for medical safety concerns, return warning if found."""
        for pattern, warning in self._medical_safety_re:
            if pattern.search(text):
                return warning
        return None

    def _is_rate_limited(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit."""
        now = time.time()
        minute_ago = now - 60

        # Clean old entries
        self._request_counts[client_id] = [
            t for t in self._request_counts[client_id] if t > minute_ago
        ]

        # Check limit
        if len(self._request_counts[client_id]) >= self.rate_limit_per_minute:
            return True

        # Record this request
        self._request_counts[client_id].append(now)
        return False


# Singleton instance
_guard: Optional[InputGuard] = None


def get_input_guard() -> InputGuard:
    """Get or create the singleton InputGuard instance."""
    global _guard
    if _guard is None:
        _guard = InputGuard()
    return _guard


def check_input(user_input: str, client_id: Optional[str] = None) -> SecurityCheckResult:
    """
    Convenience function to check input security.

    Usage:
        from backend.security import check_input
        result = check_input(user_query)
        if result.should_block:
            return {"error": result.warning_message}
    """
    return get_input_guard().check(user_input, client_id)
