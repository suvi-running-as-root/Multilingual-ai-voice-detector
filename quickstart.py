"""
Quick Start Script for AI Voice Detection API
Automatically sets up and validates the evaluation environment
"""
import subprocess
import sys
import time
import requests
import os

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def print_status(icon, message):
    print(f"{icon} {message}")

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")

    required_packages = [
        'fastapi',
        'uvicorn',
        'transformers',
        'torch',
        'librosa',
        'requests'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print_status("✅", f"{package} installed")
        except ImportError:
            print_status("❌", f"{package} NOT installed")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install -r requirements.txt")
        return False

    print_status("✅", "All dependencies installed")
    return True

def start_server():
    """Start the FastAPI server"""
    print_header("Starting API Server")

    print_status("🚀", "Launching server at http://localhost:8000")
    print_status("ℹ️", "Press Ctrl+C to stop the server")
    print("\n")

    # Start uvicorn server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print_status("\n👋", "Server stopped")

def check_server():
    """Check if server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            return True
    except:
        return False
    return False

def run_tests():
    """Run evaluation tests"""
    print_header("Running Evaluation Tests")

    # Check if server is running
    if not check_server():
        print_status("❌", "Server is not running!")
        print_status("ℹ️", "Start the server first: python quickstart.py server")
        return

    print_status("✅", "Server is running")
    print_status("🧪", "Running test suite...")
    print("\n")

    # Run test_evaluation.py
    try:
        subprocess.run([sys.executable, "test_evaluation.py"])
    except FileNotFoundError:
        print_status("❌", "test_evaluation.py not found")
    except KeyboardInterrupt:
        print_status("\n👋", "Tests interrupted")

def show_help():
    """Show help message"""
    print_header("AI Voice Detection API - Quick Start")

    print("Usage:")
    print("  python quickstart.py [command]\n")

    print("Commands:")
    print("  check      - Check if dependencies are installed")
    print("  server     - Start the API server")
    print("  test       - Run evaluation tests")
    print("  help       - Show this help message\n")

    print("Examples:")
    print("  python quickstart.py check     # Check dependencies")
    print("  python quickstart.py server    # Start server")
    print("  python quickstart.py test      # Run tests\n")

    print("Quick Setup:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Start server: python quickstart.py server")
    print("  3. In another terminal, run tests: python quickstart.py test\n")

    print("For detailed documentation, see EVALUATION_GUIDE.md")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()

    if command == "check":
        check_dependencies()

    elif command == "server":
        if not check_dependencies():
            print_status("\n❌", "Please install dependencies first")
            sys.exit(1)
        start_server()

    elif command == "test":
        run_tests()

    elif command == "help":
        show_help()

    else:
        print_status("❌", f"Unknown command: {command}")
        print_status("ℹ️", "Run 'python quickstart.py help' for usage")

if __name__ == "__main__":
    main()
