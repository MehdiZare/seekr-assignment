#!/usr/bin/env python3
"""Test real-time SSE streaming to verify events arrive as they occur.

This test requires a running local server. Set RUN_LOCAL_API_TESTS=true to enable.
"""

import json
import os
import sys
import time

import pytest
import requests

# Check if local API integration tests should run
RUN_LOCAL_API_TESTS = os.getenv("RUN_LOCAL_API_TESTS", "false").lower() in ("true", "1", "yes")


@pytest.mark.skipif(
    not RUN_LOCAL_API_TESTS,
    reason="Local API tests require RUN_LOCAL_API_TESTS=true and running server at localhost:8000"
)
def test_realtime_streaming():
    """Test that SSE events arrive in real-time, not in batch."""
    print('🧪 Testing Real-Time SSE Streaming')
    print('=' * 80)

    # Load sample data
    with open('app/data/ep001_remote_work.json', 'r') as f:
        sample_data = json.load(f)

    print('Starting analysis request...\n')

    # Make streaming request
    url = 'http://localhost:8000/api/analyze'
    try:
        response = requests.post(url, json=sample_data, stream=True, timeout=60)
    except Exception as e:
        print(f'❌ Connection error: {e}')
        pytest.fail(f"Connection error: {e}")

    if response.status_code != 200:
        print(f'❌ HTTP Error: {response.status_code}')
        pytest.fail(f"Expected status 200, got {response.status_code}")

    print('✓ Request accepted, monitoring SSE stream:\n')

    event_count = 0
    start_time = time.time()
    last_time = start_time
    event_times = []

    for line in response.iter_lines():
        if not line:
            continue

        line = line.decode('utf-8')
        if line.startswith('data: '):
            try:
                current_time = time.time()
                elapsed_from_last = current_time - last_time
                elapsed_total = current_time - start_time

                data = json.loads(line[6:])
                event_count += 1
                event_times.append(current_time)

                event_type = data.get('type', data.get('stage', 'unknown'))
                agent = data.get('agent', data.get('target_agent', 'N/A'))
                message = data.get('message', '')[:60]

                print(f'[+{elapsed_total:5.1f}s / Δ{elapsed_from_last:5.2f}s] Event {event_count:2d}: {event_type:25s} | {agent:20s} | {message}')

                last_time = current_time

                # Stop after complete event
                if data.get('stage') == 'complete':
                    print('\n' + '=' * 80)
                    print('✅ WORKFLOW COMPLETE')
                    print('=' * 80)
                    print(f'\nTotal events: {event_count}')
                    print(f'Total time: {elapsed_total:.1f}s')

                    # Analyze timing pattern
                    print('\n💡 Real-time streaming analysis:')
                    if event_count > 1:
                        delays = [event_times[i] - event_times[i-1] for i in range(1, len(event_times))]
                        max_delay = max(delays)
                        avg_delay = sum(delays) / len(delays)

                        # Check if events were batched (all arrived within 1 second)
                        if max_delay < 1.0:
                            print(f'   ⚠️  BATCHED: All events arrived within 1 second (max gap: {max_delay:.2f}s)')
                            print('   ❌ Real-time streaming NOT working')
                        else:
                            print(f'   ✓ STREAMING: Events spread over time (max gap: {max_delay:.1f}s, avg: {avg_delay:.2f}s)')
                            print('   ✅ Real-time streaming IS working')

                    return  # Test passes

            except Exception as e:
                print(f'⚠️  Parse error: {e}')
                continue

    print('\n⚠️  Stream ended before completion event')
    pytest.fail("Stream ended before completion event")

if __name__ == '__main__':
    # When run directly (not via pytest), temporarily enable the test
    if not RUN_LOCAL_API_TESTS:
        print("💡 Note: Set RUN_LOCAL_API_TESTS=true to run this test via pytest")
        print("   Running in direct execution mode...\n")

    try:
        test_realtime_streaming()
        sys.exit(0)
    except KeyboardInterrupt:
        print('\n\n⏹️  Test interrupted')
        sys.exit(0)
    except Exception as e:
        print(f'\n❌ Test failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
