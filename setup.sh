#!/bin/bash

echo "🚀 AI-Scale PoC - Setup & Launch"
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    echo "⬆️  Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    
    echo "✅ Setup complete!"
else
    echo "♻️  Using existing virtual environment..."
    source venv/bin/activate
fi

# Launch the application
echo ""
echo "🎯 Launching AI-Scale POS Application..."
echo "========================================="
python aiscale_pos_gui.py