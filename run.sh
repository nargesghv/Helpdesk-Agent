#!/bin/bash
# Run Helpdesk Agent Backend

# Check if GROQ_API_KEY is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  WARNING: GROQ_API_KEY not set!"
    echo "Set it with: export GROQ_API_KEY='your-key'"
    echo ""
fi

echo "🚀 Starting Helpdesk Agent on http://localhost:8004"
echo "📱 Open browser: http://localhost:8004"
echo ""

# Run with uvicorn
uvicorn backend:app --host 0.0.0.0 --port 8004 --reload

