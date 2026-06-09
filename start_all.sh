#!/bin/bash
# Start all Legal Multi-Agent System services
# Registry must be first, then leaf agents, then orchestrators

# Clean up existing processes on the target ports to avoid "port already in use" errors
echo "Cleaning up any existing processes on ports 10000, 10100, 10101, 10102, 10103..."
for port in 10000 10100 10101 10102 10103; do
  pid=$(lsof -t -i:$port 2>/dev/null)
  if [ -n "$pid" ]; then
    echo "Killing existing process on port $port (PID: $pid)"
    kill -9 $pid 2>/dev/null || true
  fi
done

# Trap exit/SIGINT/SIGTERM to kill all background services when this script exits
trap 'echo "Stopping all services..."; kill $(jobs -p) 2>/dev/null || true; exit' INT TERM EXIT

echo "Starting Registry service on port 10000..."
python3 -m registry &
sleep 2

echo "Starting Tax Agent on port 10102..."
python3 -m tax_agent &

echo "Starting Compliance Agent on port 10103..."
python3 -m compliance_agent &
sleep 3

echo "Starting Law Agent on port 10101..."
python3 -m law_agent &
sleep 3

echo "Starting Customer Agent on port 10100..."
python3 -m customer_agent &

echo ""
echo "All services started:"
echo "  Registry:         http://localhost:10000"
echo "  Customer Agent:   http://localhost:10100"
echo "  Law Agent:        http://localhost:10101"
echo "  Tax Agent:        http://localhost:10102"
echo "  Compliance Agent: http://localhost:10103"
echo ""
echo "Run test_client.py to send a query:"
echo "  python3 test_client.py"
echo ""
echo "Press Ctrl+C to stop all services."

# Wait for all background processes
wait