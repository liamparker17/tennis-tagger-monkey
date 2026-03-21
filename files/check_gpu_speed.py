"""
GPU Speed Check - Diagnose why detection is slow

Run this to check:
1. Is CUDA available?
2. Is YOLO using GPU?
3. What's the actual inference speed?
"""

import sys
import time
import numpy as np
from pathlib import Path

print("=" * 70)
print("GPU SPEED CHECK")
print("=" * 70)
print()

# Check 1: PyTorch CUDA
print("1. Checking PyTorch CUDA...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__} installed")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA available: {cuda_available}")

    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
    else:
        print("   ⚠ WARNING: CUDA not available - will use CPU (very slow)")
        print("   Fix: pip uninstall torch torchvision")
        print("        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
except ImportError as e:
    print(f"   ✗ PyTorch not installed: {e}")
    sys.exit(1)

print()

# Check 2: YOLO Model
print("2. Checking YOLO model...")
try:
    from ultralytics import YOLO

    model_path = Path("models/yolov8x.pt")
    if not model_path.exists():
        print(f"   ✗ Model not found: {model_path}")
        sys.exit(1)

    print(f"   Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Check which device model is on
    device = next(model.model.parameters()).device
    print(f"   Model device: {device}")

    if str(device) == 'cpu' and cuda_available:
        print("   ⚠ WARNING: Model on CPU despite CUDA being available")
        print("   This will be VERY slow!")

except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    sys.exit(1)

print()

# Check 3: Single Frame Inference Speed
print("3. Testing single frame inference...")
test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

# Warmup run
print("   Running warmup...")
_ = model.predict(test_frame, verbose=False)

# Timed run
print("   Running timed inference...")
times = []
for i in range(10):
    start = time.time()
    results = model.predict(test_frame, verbose=False, conf=0.25)
    end = time.time()
    times.append((end - start) * 1000)

avg_time = np.mean(times)
std_time = np.std(times)

print(f"   Average time: {avg_time:.1f}ms ± {std_time:.1f}ms")

if cuda_available:
    if avg_time < 100:
        print("   ✓ EXCELLENT - Using GPU efficiently")
    elif avg_time < 300:
        print("   ⚠ SLOW - GPU may be throttling or model too large")
    else:
        print("   ✗ VERY SLOW - Likely not using GPU despite CUDA available")
else:
    if avg_time < 1000:
        print("   ✓ OK for CPU")
    else:
        print("   ✗ VERY SLOW - Even for CPU this is slow")

print()

# Check 4: Batch Inference Speed
print("4. Testing batch inference (32 frames)...")
batch_frames = [test_frame for _ in range(32)]

# Warmup
_ = model.predict(batch_frames, verbose=False)

# Timed run
start = time.time()
results = model.predict(batch_frames, verbose=False, conf=0.25)
end = time.time()

batch_time = (end - start) * 1000
per_frame = batch_time / 32

print(f"   Batch time: {batch_time:.0f}ms")
print(f"   Per frame: {per_frame:.1f}ms")

if cuda_available:
    if batch_time < 2000:
        print("   ✓ EXCELLENT - Batch processing working well")
    elif batch_time < 5000:
        print("   ⚠ ACCEPTABLE - Could be faster")
    else:
        print(f"   ✗ TOO SLOW - {batch_time/1000:.1f} seconds for 32 frames")
        print("   Expected: <2 seconds on GPU")
        print("   Your issue: This is your bottleneck!")
else:
    if batch_time < 20000:
        print("   ✓ EXPECTED for CPU")
    else:
        print("   ✗ TOO SLOW even for CPU")

print()

# Check 5: Model Size Recommendation
print("5. Model optimization recommendations...")
model_size_mb = model_path.stat().st_size / (1024**2)
print(f"   Current model: yolov8x.pt ({model_size_mb:.1f} MB)")

if avg_time > 200 or batch_time > 3000:
    print("   ⚠ RECOMMENDATION: Use smaller model")
    print("   ")
    print("   Speed comparison:")
    print("   - yolov8x.pt (136 MB): Slowest, most accurate")
    print("   - yolov8m.pt (52 MB):  2-3x faster, 95% accuracy")
    print("   - yolov8s.pt (22 MB):  4-5x faster, 90% accuracy")
    print("   ")
    print("   Change config\\config.yaml:")
    print('   model: "yolov8m.pt"  # Instead of yolov8x.pt')

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

if cuda_available and avg_time < 100 and batch_time < 2000:
    print("✓ System is properly configured and fast")
    print(f"  Expected processing speed: {32 / (batch_time/1000):.1f} FPS")
elif not cuda_available:
    print("✗ PROBLEM: No CUDA support")
    print("  ACTION: Reinstall PyTorch with CUDA")
    print("  Command: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
elif cuda_available and avg_time > 500:
    print("✗ PROBLEM: CUDA available but not being used")
    print("  ACTION: Check GPU drivers and PyTorch installation")
else:
    print("⚠ SUBOPTIMAL: Using GPU but slower than expected")
    print(f"  Current speed: {batch_time/1000:.1f} sec per 32 frames")
    print(f"  Expected speed: <2 sec per 32 frames")
    print("  ACTION: Switch to smaller model (yolov8m.pt)")

print()
