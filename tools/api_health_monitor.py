#!/usr/bin/env python3
"""
API Health Monitor for RSS Analyzer

This script tests all available APIs and provides detailed status reports
to help diagnose which APIs are failing and why.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from claude_client import ClaudeClient
    from mistral_client import MistralClient  
    from openai_client import OpenAIClient
    from utils import load_config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


@dataclass
class APITestResult:
    """Result of an API test"""
    provider: str
    status: str  # "success", "failed", "no_key", "unavailable"
    response_time_ms: Optional[float]
    error_message: Optional[str]
    error_type: Optional[str]
    http_status: Optional[int]
    model_used: Optional[str]
    credits_info: Optional[str]
    timestamp: str


@dataclass
class MonitoringReport:
    """Complete monitoring report"""
    timestamp: str
    total_apis: int
    working_apis: int
    failed_apis: int
    apis_without_keys: int
    recommended_provider: Optional[str]
    results: List[APITestResult]


class APIHealthMonitor:
    """Monitor health of all API providers"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the monitor"""
        logger = logging.getLogger("api_monitor")
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # File handler for API monitoring logs
        file_handler = logging.FileHandler(logs_dir / "api_health.log")
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    async def test_anthropic_api(self) -> APITestResult:
        """Test Anthropic Claude API"""
        start_time = datetime.now()
        
        try:
            # Check for API key
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return APITestResult(
                    provider="anthropic",
                    status="no_key",
                    response_time_ms=None,
                    error_message="ANTHROPIC_API_KEY environment variable not set",
                    error_type="missing_credentials",
                    http_status=None,
                    model_used=self.config.get("api", {}).get("anthropic", {}).get("model"),
                    credits_info=None,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            
            self.logger.info("Testing Anthropic Claude API...")
            
            # Initialize client
            client = ClaudeClient(self.config)
            
            # Test the API with a simple message
            start_test = datetime.now()
            test_response = await asyncio.to_thread(client.test_connection)
            end_test = datetime.now()
            
            response_time = (end_test - start_test).total_seconds() * 1000
            
            if test_response:
                self.logger.info("✅ Anthropic API: Connection successful")
                return APITestResult(
                    provider="anthropic",
                    status="success",
                    response_time_ms=response_time,
                    error_message=None,
                    error_type=None,
                    http_status=200,
                    model_used=self.config.get("api", {}).get("anthropic", {}).get("model"),
                    credits_info="Available",
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            else:
                return APITestResult(
                    provider="anthropic",
                    status="failed",
                    response_time_ms=response_time,
                    error_message="API test returned False",
                    error_type="unknown_failure",
                    http_status=None,
                    model_used=self.config.get("api", {}).get("anthropic", {}).get("model"),
                    credits_info="Unknown",
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                
        except Exception as e:
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            error_msg = str(e)
            error_type = type(e).__name__
            http_status = None
            credits_info = None
            
            # Parse specific error types
            if "credit balance" in error_msg.lower():
                error_type = "insufficient_credits"
                credits_info = "Insufficient"
                http_status = 400
            elif "rate limit" in error_msg.lower():
                error_type = "rate_limited"
                http_status = 429
            elif "invalid" in error_msg.lower() and "key" in error_msg.lower():
                error_type = "invalid_key"
                http_status = 401
            elif "400" in error_msg:
                http_status = 400
            elif "401" in error_msg:
                http_status = 401
            elif "429" in error_msg:
                http_status = 429
                error_type = "rate_limited"
            elif "500" in error_msg:
                http_status = 500
                error_type = "server_error"
            
            self.logger.error(f"❌ Anthropic API failed: {error_msg}")
            
            return APITestResult(
                provider="anthropic",
                status="failed",
                response_time_ms=response_time,
                error_message=error_msg,
                error_type=error_type,
                http_status=http_status,
                model_used=self.config.get("api", {}).get("anthropic", {}).get("model"),
                credits_info=credits_info,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    async def test_mistral_api(self) -> APITestResult:
        """Test Mistral AI API"""
        start_time = datetime.now()
        
        try:
            # Check for API key
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                return APITestResult(
                    provider="mistral",
                    status="no_key",
                    response_time_ms=None,
                    error_message="MISTRAL_API_KEY environment variable not set",
                    error_type="missing_credentials",
                    http_status=None,
                    model_used=self.config.get("api", {}).get("mistral", {}).get("model"),
                    credits_info=None,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            
            self.logger.info("Testing Mistral AI API...")
            
            # Initialize client
            client = MistralClient(self.config)
            
            # Test the API
            start_test = datetime.now()
            test_response = await asyncio.to_thread(client.test_connection)
            end_test = datetime.now()
            
            response_time = (end_test - start_test).total_seconds() * 1000
            
            if test_response:
                self.logger.info("✅ Mistral API: Connection successful")
                return APITestResult(
                    provider="mistral",
                    status="success",
                    response_time_ms=response_time,
                    error_message=None,
                    error_type=None,
                    http_status=200,
                    model_used=self.config.get("api", {}).get("mistral", {}).get("model"),
                    credits_info="Available",
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            else:
                return APITestResult(
                    provider="mistral",
                    status="failed",
                    response_time_ms=response_time,
                    error_message="API test returned False",
                    error_type="unknown_failure",
                    http_status=None,
                    model_used=self.config.get("api", {}).get("mistral", {}).get("model"),
                    credits_info="Unknown",
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                
        except Exception as e:
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            error_msg = str(e)
            error_type = type(e).__name__
            http_status = None
            credits_info = None
            
            # Parse specific error types
            if "capacity exceeded" in error_msg.lower():
                error_type = "capacity_exceeded"
                http_status = 429
            elif "rate limit" in error_msg.lower():
                error_type = "rate_limited"
                http_status = 429
            elif "invalid" in error_msg.lower() and "key" in error_msg.lower():
                error_type = "invalid_key"
                http_status = 401
            elif "429" in error_msg:
                http_status = 429
                error_type = "rate_limited"
            elif "401" in error_msg:
                http_status = 401
            elif "500" in error_msg:
                http_status = 500
                error_type = "server_error"
            
            self.logger.error(f"❌ Mistral API failed: {error_msg}")
            
            return APITestResult(
                provider="mistral",
                status="failed",
                response_time_ms=response_time,
                error_message=error_msg,
                error_type=error_type,
                http_status=http_status,
                model_used=self.config.get("api", {}).get("mistral", {}).get("model"),
                credits_info=credits_info,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    async def test_openai_api(self) -> APITestResult:
        """Test OpenAI API"""
        start_time = datetime.now()
        
        try:
            # Check for API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return APITestResult(
                    provider="openai",
                    status="no_key",
                    response_time_ms=None,
                    error_message="OPENAI_API_KEY environment variable not set",
                    error_type="missing_credentials",
                    http_status=None,
                    model_used=self.config.get("api", {}).get("openai", {}).get("model"),
                    credits_info=None,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            
            self.logger.info("Testing OpenAI API...")
            
            # Initialize client
            client = OpenAIClient(self.config)
            
            # Test the API
            start_test = datetime.now()
            test_response = await asyncio.to_thread(client.test_connection)
            end_test = datetime.now()
            
            response_time = (end_test - start_test).total_seconds() * 1000
            
            if test_response:
                self.logger.info("✅ OpenAI API: Connection successful")
                return APITestResult(
                    provider="openai",
                    status="success",
                    response_time_ms=response_time,
                    error_message=None,
                    error_type=None,
                    http_status=200,
                    model_used=self.config.get("api", {}).get("openai", {}).get("model"),
                    credits_info="Available",
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            else:
                return APITestResult(
                    provider="openai",
                    status="failed",
                    response_time_ms=response_time,
                    error_message="API test returned False",
                    error_type="unknown_failure",
                    http_status=None,
                    model_used=self.config.get("api", {}).get("openai", {}).get("model"),
                    credits_info="Unknown",
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                
        except Exception as e:
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            error_msg = str(e)
            error_type = type(e).__name__
            http_status = None
            credits_info = None
            
            # Parse specific error types
            if "insufficient" in error_msg.lower() and "quota" in error_msg.lower():
                error_type = "insufficient_quota"
                credits_info = "Insufficient"
                http_status = 429
            elif "rate limit" in error_msg.lower():
                error_type = "rate_limited"
                http_status = 429
            elif "invalid" in error_msg.lower() and "key" in error_msg.lower():
                error_type = "invalid_key"
                http_status = 401
            elif "429" in error_msg:
                http_status = 429
                error_type = "rate_limited"
            elif "401" in error_msg:
                http_status = 401
            elif "500" in error_msg:
                http_status = 500
                error_type = "server_error"
            
            self.logger.error(f"❌ OpenAI API failed: {error_msg}")
            
            return APITestResult(
                provider="openai",
                status="failed",
                response_time_ms=response_time,
                error_message=error_msg,
                error_type=error_type,
                http_status=http_status,
                model_used=self.config.get("api", {}).get("openai", {}).get("model"),
                credits_info=credits_info,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    async def run_health_check(self) -> MonitoringReport:
        """Run health check on all APIs"""
        self.logger.info("🔍 Starting API health check...")
        
        # Test all APIs concurrently
        tasks = [
            self.test_anthropic_api(),
            self.test_mistral_api(),
            self.test_openai_api()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in the results
        api_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Unexpected error during API test: {result}")
                # Create a generic failure result
                api_results.append(APITestResult(
                    provider="unknown",
                    status="failed",
                    response_time_ms=None,
                    error_message=str(result),
                    error_type="unexpected_error",
                    http_status=None,
                    model_used=None,
                    credits_info=None,
                    timestamp=datetime.now(timezone.utc).isoformat()
                ))
            else:
                api_results.append(result)
        
        # Analyze results
        working_apis = sum(1 for r in api_results if r.status == "success")
        failed_apis = sum(1 for r in api_results if r.status == "failed")
        apis_without_keys = sum(1 for r in api_results if r.status == "no_key")
        
        # Recommend best provider
        working_providers = [r for r in api_results if r.status == "success"]
        recommended_provider = None
        if working_providers:
            # Sort by response time (fastest first)
            working_providers.sort(key=lambda x: x.response_time_ms or float('inf'))
            recommended_provider = working_providers[0].provider
        
        report = MonitoringReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_apis=len(api_results),
            working_apis=working_apis,
            failed_apis=failed_apis,
            apis_without_keys=apis_without_keys,
            recommended_provider=recommended_provider,
            results=api_results
        )
        
        self.logger.info(f"✅ Health check complete: {working_apis}/{len(api_results)} APIs working")
        return report
    
    def save_report(self, report: MonitoringReport, output_file: str = "logs/api_health_report.json"):
        """Save monitoring report to file"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(asdict(report), f, indent=2)
            
            self.logger.info(f"📄 Report saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
    
    def print_summary(self, report: MonitoringReport):
        """Print a human-readable summary"""
        print(f"\n{'='*60}")
        print("🏥 API HEALTH MONITORING REPORT")
        print(f"{'='*60}")
        print(f"⏰ Timestamp: {report.timestamp}")
        print(f"📊 APIs Tested: {report.total_apis}")
        print(f"✅ Working: {report.working_apis}")
        print(f"❌ Failed: {report.failed_apis}")
        print(f"🔑 Missing Keys: {report.apis_without_keys}")
        
        if report.recommended_provider:
            print(f"⭐ Recommended: {report.recommended_provider}")
        else:
            print("⚠️  No working APIs found!")
        
        print(f"\n{'─'*60}")
        print("📋 DETAILED RESULTS")
        print(f"{'─'*60}")
        
        for result in report.results:
            status_emoji = "✅" if result.status == "success" else "❌" if result.status == "failed" else "🔑"
            
            print(f"\n{status_emoji} {result.provider.upper()}")
            print(f"   Status: {result.status}")
            print(f"   Model: {result.model_used or 'Unknown'}")
            
            if result.response_time_ms:
                print(f"   Response Time: {result.response_time_ms:.0f}ms")
            
            if result.error_message:
                print(f"   Error: {result.error_message}")
            
            if result.error_type:
                print(f"   Error Type: {result.error_type}")
            
            if result.http_status:
                print(f"   HTTP Status: {result.http_status}")
            
            if result.credits_info:
                print(f"   Credits: {result.credits_info}")
        
        print(f"\n{'='*60}")


async def main():
    """Main function"""
    monitor = APIHealthMonitor()
    
    try:
        # Run health check
        report = await monitor.run_health_check()
        
        # Save report
        monitor.save_report(report)
        
        # Print summary
        monitor.print_summary(report)
        
        # Exit with appropriate code
        if report.working_apis > 0:
            sys.exit(0)  # Success - at least one API is working
        else:
            sys.exit(1)  # Failure - no APIs are working
            
    except KeyboardInterrupt:
        print("\n🛑 Health check interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())