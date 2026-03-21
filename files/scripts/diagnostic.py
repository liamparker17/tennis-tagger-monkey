"""
Diagnostic Script
Tests system configuration and identifies issues
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def check_python():
    """Check Python version"""
    print_header("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 9 <= version.minor <= 11:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python version should be 3.9-3.11")
        return False

def check_package(package_name, import_name=None):
    """Check if package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} installed")
        return True
    except ImportError:
        print(f"✗ {package_name} not installed")
        return False

def check_packages():
    """Check required packages"""
    print_header("Required Packages")
    
    packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('opencv-python', 'cv2'),
        ('PyTorch', 'torch'),
        ('torchvision', 'torchvision'),
        ('ultralytics', 'ultralytics'),
        ('mediapipe', 'mediapipe'),
        ('scikit-learn', 'sklearn'),
        ('PyYAML', 'yaml'),
        ('tqdm', 'tqdm'),
        ('PyQt5', 'PyQt5'),
    ]
    
    results = []
    for package, import_name in packages:
        results.append(check_package(package, import_name))
    
    return all(results)

def check_cuda():
    """Check CUDA availability"""
    print_header("CUDA/GPU Support")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            
            return True
        else:
            print("✗ CUDA not available (CPU-only mode)")
            print("  To enable GPU:")
            print("  1. Install NVIDIA drivers")
            print("  2. Install CUDA Toolkit 11.8+")
            print("  3. Reinstall PyTorch with CUDA support")
            return False
            
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_ffmpeg():
    """Check FFmpeg installation"""
    print_header("FFmpeg")
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ FFmpeg installed: {version_line}")
            return True
        else:
            print("✗ FFmpeg not working properly")
            return False
            
    except FileNotFoundError:
        print("✗ FFmpeg not installed")
        print("  Install with:")
        print("    Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("    Mac: brew install ffmpeg")
        print("    Windows: Download from ffmpeg.org")
        return False
    except Exception as e:
        print(f"✗ Error checking FFmpeg: {e}")
        return False

def check_models():
    """Check if models are downloaded"""
    print_header("Pre-trained Models")
    
    models_dir = Path('models')
    
    if not models_dir.exists():
        print("✗ Models directory not found")
        print("  Run: python scripts/download_models.py")
        return False
    
    required_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 
                      'yolov8l.pt', 'yolov8x.pt']
    
    found = []
    missing = []
    
    for model in required_models:
        model_path = models_dir / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✓ {model} ({size_mb:.1f} MB)")
            found.append(model)
        else:
            print(f"✗ {model} not found")
            missing.append(model)
    
    if missing:
        print(f"\nMissing {len(missing)} models. Run: python scripts/download_models.py")
        return False
    
    return True

def check_config():
    """Check configuration file"""
    print_header("Configuration")
    
    config_path = Path('config/config.yaml')
    
    if not config_path.exists():
        print("✗ Configuration file not found: config/config.yaml")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Configuration file is valid")
        
        # Check key settings
        if config.get('hardware', {}).get('use_gpu'):
            print("  GPU acceleration: ENABLED")
        else:
            print("  GPU acceleration: DISABLED")
        
        batch_size = config.get('hardware', {}).get('batch_size', 8)
        print(f"  Batch size: {batch_size}")
        
        model = config.get('detection', {}).get('player_detector', {}).get('model', 'unknown')
        print(f"  Detection model: {model}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading configuration: {e}")
        return False

def check_directories():
    """Check directory structure"""
    print_header("Directory Structure")
    
    required_dirs = [
        'src',
        'src/detection',
        'src/analysis',
        'models',
        'data',
        'data/training',
        'data/inference',
        'data/output',
        'scripts',
        'config',
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} not found")
            all_exist = False
    
    return all_exist

def main():
    """Run full diagnostic"""
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#" + " " * 15 + "TENNIS TAGGER DIAGNOSTIC" + " " * 19 + "#")
    print("#" + " " * 58 + "#")
    print("#" * 60)
    
    results = {
        'Python Version': check_python(),
        'Required Packages': check_packages(),
        'CUDA/GPU': check_cuda(),
        'FFmpeg': check_ffmpeg(),
        'Pre-trained Models': check_models(),
        'Configuration': check_config(),
        'Directory Structure': check_directories(),
    }
    
    # Summary
    print_header("DIAGNOSTIC SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check:25s} {status}")
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ System is ready to use!")
    else:
        print("\n✗ Some issues found. See details above.")
        print("\nQuick fixes:")
        
        if not results['Required Packages']:
            print("  - Install packages: pip install -r requirements.txt")
        
        if not results['Pre-trained Models']:
            print("  - Download models: python scripts/download_models.py")
        
        if not results['FFmpeg']:
            print("  - Install FFmpeg (see instructions above)")
        
        if not results['Directory Structure']:
            print("  - Ensure you're in the project root directory")

if __name__ == "__main__":
    main()
