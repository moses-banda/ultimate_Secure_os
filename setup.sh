#!/bin/bash

# FaceAuth System - Automated Setup Script
# This script sets up the entire system automatically

set -e  # Exit on error

echo "======================================"
echo "   FaceAuth System Setup"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check Python version
print_info "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check pip
print_info "Checking pip installation..."
if command -v pip &> /dev/null || command -v pip3 &> /dev/null; then
    print_success "pip found"
else
    print_error "pip not found. Installing..."
    python3 -m ensurepip --default-pip
fi

# Installation menu
echo ""
echo "Select installation type:"
echo "  1. Quick Demo (gesture detection only)"
echo "  2. Full System (with face recognition)"
echo "  3. Full System + AWS setup"
echo ""
read -p "Enter choice (1-3): " INSTALL_TYPE

case $INSTALL_TYPE in
    1)
        print_info "Installing minimal dependencies for demo..."
        pip3 install opencv-python mediapipe numpy --break-system-packages || \
            pip install opencv-python mediapipe numpy
        print_success "Demo dependencies installed!"
        
        echo ""
        print_info "Running gesture detection demo..."
        python3 demo.py
        ;;
        
    2)
        print_info "Installing full system dependencies..."
        pip3 install -r requirements.txt --break-system-packages || \
            pip install -r requirements.txt
        print_success "Full dependencies installed!"
        
        # Create training data directory
        if [ ! -d "training_data" ]; then
            print_info "Creating training_data directory..."
            mkdir -p training_data/sample_user
            print_success "Training data directory created"
            echo ""
            echo "Add your photos to: training_data/your_name/"
            echo "  - At least 3-5 photos"
            echo "  - Different angles"
            echo "  - Good lighting"
        fi
        
        echo ""
        print_success "Setup complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Add training photos to training_data/"
        echo "  2. Run: python3 enroll_users.py --training-dir ./training_data"
        echo "  3. Test: python3 demo.py"
        ;;
        
    3)
        print_info "Installing full system with AWS..."
        pip3 install -r requirements.txt --break-system-packages || \
            pip install -r requirements.txt
        print_success "Dependencies installed!"
        
        # Check AWS CLI
        print_info "Checking AWS CLI..."
        if command -v aws &> /dev/null; then
            print_success "AWS CLI found"
            
            # Configure AWS
            echo ""
            print_info "Configuring AWS credentials..."
            aws configure
            
            # Create DynamoDB tables
            print_info "Creating DynamoDB tables..."
            python3 -c "
from facial_auth_system import FacialEmbeddingSystem
auth = FacialEmbeddingSystem()
auth.create_aws_tables()
print('Tables created!')
"
            print_success "AWS setup complete!"
            
        else
            print_error "AWS CLI not found"
            echo "Install from: https://aws.amazon.com/cli/"
            echo "Then run: aws configure"
        fi
        
        # Create training data directory
        if [ ! -d "training_data" ]; then
            mkdir -p training_data/sample_user
            print_success "Training data directory created"
        fi
        
        echo ""
        print_success "Full setup complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Add training photos to training_data/"
        echo "  2. Run: python3 enroll_users.py --training-dir ./training_data"
        echo "  3. Users will be enrolled to AWS DynamoDB"
        ;;
        
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "   Setup Complete! 🎉"
echo "======================================"
echo ""
echo "Quick start commands:"
echo "  • Demo:      python3 demo.py"
echo "  • Enroll:    python3 enroll_users.py --training-dir ./training_data"
echo "  • Test:      python3 enroll_users.py --test-image face.jpg"
echo ""
echo "Read QUICKSTART.md for detailed instructions"
echo ""