#!/usr/bin/env python3
"""
Simple runner script for the LangChain agent evaluation.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def run_quick_evaluation():
    """Run a subset of critical tests for quick feedback"""
    from evaluation import HealthDataGenerator, AgentEvaluator
    
    print("🚀 Running Quick Evaluation (Core Tests Only)")
    print("-" * 50)
    
    # Generate smaller dataset for quick tests
    data_generator = HealthDataGenerator(days=30)
    evaluator = AgentEvaluator()
    
    # Generate data and setup environment
    sample_data = data_generator.generate_sample_data()
    if not evaluator.setup_test_environment(sample_data):
        print("❌ Failed to setup test environment")
        return
    
    # Generate only critical test cases
    all_test_cases = evaluator.generate_test_cases(sample_data)
    critical_tests = [tc for tc in all_test_cases if tc.bug_category in [
        'data_accuracy', 'visualization_failure', 'context_handling'
    ]]
    
    print(f"Running {len(critical_tests)} critical tests...")
    
    # Run evaluation
    results = evaluator.run_evaluation(critical_tests)
    
    if not results:
        print("❌ No test results obtained - check agent initialization")
        return
        
    report = evaluator.generate_report(results)
    
    if "error" in report:
        print(f"❌ Error generating report: {report['error']}")
        return
    
    # Quick summary
    summary = report["summary"]
    print(f"\n📊 QUICK RESULTS:")
    print(f"   Success Rate: {summary['success_rate']}")
    print(f"   Time: {summary['avg_execution_time']}")
    
    if report["failed_tests"]:
        print(f"   ❌ Failed: {', '.join([t['name'] for t in report['failed_tests']])}")
    else:
        print("   ✅ All critical tests passed!")

def run_full_evaluation():
    """Run the complete evaluation suite"""
    from evaluation import main
    return main()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LangChain Agent Evaluation")
    parser.add_argument("--quick", action="store_true", 
                       help="Run only critical tests for quick feedback")
    parser.add_argument("--full", action="store_true", 
                       help="Run complete evaluation suite")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_evaluation()
    elif args.full or not any([args.quick]):
        run_full_evaluation()
    else:
        print("Use --quick for fast tests or --full for complete evaluation")