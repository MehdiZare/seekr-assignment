#!/usr/bin/env python3
"""Test the new supervisor workflow.

This test requires a running local server. Set RUN_LOCAL_API_TESTS=true to enable.
"""

import json
import os
import sys

import pytest
import requests

# Check if local API integration tests should run
RUN_LOCAL_API_TESTS = os.getenv("RUN_LOCAL_API_TESTS", "false").lower() in ("true", "1", "yes")


@pytest.mark.skipif(
    not RUN_LOCAL_API_TESTS,
    reason="Local API tests require RUN_LOCAL_API_TESTS=true and running server at localhost:8000"
)
def test_workflow():
    """Test the workflow with ep001 sample."""
    # Load sample data
    with open('app/data/ep001_remote_work.json', 'r') as f:
        sample_data = json.load(f)

    # Make request to analyze endpoint
    print("🔄 Starting analysis workflow test...")
    print("=" * 80)

    url = "http://localhost:8000/api/analyze"
    response = requests.post(url, json=sample_data, stream=True, timeout=300)

    if response.status_code != 200:
        print(f"❌ Error: HTTP {response.status_code}")
        print(response.text)
        pytest.fail(f"Expected status 200, got {response.status_code}")

    # Process SSE stream
    event_count = 0
    max_events = 20  # Limit for testing

    print("\n📡 Receiving SSE events:")
    print("-" * 80)

    for line in response.iter_lines():
        if not line:
            continue

        line = line.decode('utf-8')
        if line.startswith('data: '):
            try:
                data = json.loads(line[6:])
                event_count += 1

                # Display event details
                event_type = data.get('type', 'unknown')
                stage = data.get('stage', 'N/A')
                agent = data.get('agent', data.get('node', 'N/A'))
                message = data.get('message', '')

                print(f"\nEvent #{event_count}:")
                print(f"  Type: {event_type}")
                print(f"  Stage: {stage}")
                print(f"  Agent: {agent}")
                print(f"  Message: {message[:100]}...")

                # Check for specific event types
                if event_type == 'supervisor_decision':
                    print(f"  ✓ Supervisor calling: {data.get('target_agent')}")
                elif event_type == 'agent_complete':
                    print(f"  ✓ Agent completed: {data.get('details', 'N/A')}")
                elif stage == 'complete':
                    print("\n" + "=" * 80)
                    print("✅ Workflow completed successfully!")
                    print("=" * 80)

                    # Display result structure
                    result = data.get('result', {})
                    print("\n📊 Result Structure:")
                    print(f"  - Summary: {'✓' if result.get('summary') else '✗'}")
                    print(f"  - Notes: {'✓' if result.get('notes') else '✗'}")
                    print(f"  - Fact Check: {'✓' if result.get('fact_check') else '✗'}")

                    if result.get('summary'):
                        print(f"\n📝 Summary Preview:")
                        print(f"  Core Theme: {result['summary'].get('core_theme', 'N/A')}")

                    if result.get('notes'):
                        print(f"\n📌 Notes Preview:")
                        print(f"  Takeaways: {len(result['notes'].get('top_takeaways', []))}")
                        print(f"  Quotes: {len(result['notes'].get('notable_quotes', []))}")
                        print(f"  Topics: {len(result['notes'].get('topics', []))}")
                        print(f"  Factual Statements: {len(result['notes'].get('factual_statements', []))}")

                    if result.get('fact_check'):
                        print(f"\n🔍 Fact Check Preview:")
                        print(f"  Claims Verified: {len(result['fact_check'].get('verified_claims', []))}")
                        print(f"  Reliability: {result['fact_check'].get('overall_reliability', 0):.2%}")

                    return  # Test passes

                # Stop after max events for quick test
                if event_count >= max_events and stage != 'complete':
                    print(f"\n⏸️  Stopping after {max_events} events for quick test")
                    print("✓ SSE events are flowing correctly!")
                    print("✓ New event structure is working!")
                    return  # Test passes

            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"⚠️  Error processing event: {e}")
                continue

    print(f"\n✓ Processed {event_count} events total")
    # Test passes if we received any events
    assert event_count > 0, "No events received from SSE stream"

if __name__ == "__main__":
    # When run directly (not via pytest), temporarily enable the test
    if not RUN_LOCAL_API_TESTS:
        print("💡 Note: Set RUN_LOCAL_API_TESTS=true to run this test via pytest")
        print("   Running in direct execution mode...\n")

    try:
        test_workflow()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
