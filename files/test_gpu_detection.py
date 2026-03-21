"""
Quick test to verify GPU is being used by YOLO
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add detection to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("TESTING GPU DETECTION")
print("=" * 70)

# Test 1: Check PyTorch CUDA
print("\n1. Checking PyTorch CUDA...")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("   ERROR: CUDA not available!")
        sys.exit(1)
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# Test 2: Load YOLO and check device
print("\n2. Loading YOLO model...")
try:
    from ultralytics import YOLO

    model_path = Path("models/yolov8x.pt")
    if not model_path.exists():
        print(f"   ERROR: Model not found: {model_path}")
        sys.exit(1)

    print(f"   Loading: {model_path}")
    model = YOLO(str(model_path))

    # Force to GPU
    model.to('cuda')

    # Check device
    device = next(model.model.parameters()).device
    print(f"   Model on device: {device}")

    if str(device) == 'cpu':
        print("   ERROR: Model on CPU!")
        sys.exit(1)

except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Run detection and time it
print("\n3. Testing detection speed...")
test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

# Warmup
print("   Warmup run...")
_ = model.predict(test_frame, verbose=False, device='0')

# Single frame test
print("   Testing single frame (10 runs)...")
times = []
for i in range(10):
    start = time.time()
    results = model.predict(test_frame, verbose=False, conf=0.25, device='0')
    end = time.time()
    times.append((end - start) * 1000)

avg_time = np.mean(times)
print(f"   Average: {avg_time:.1f}ms per frame")

if avg_time < 100:
    print("   ✓ EXCELLENT - GPU is being used!")
elif avg_time < 300:
    print("   ⚠ SLOW - GPU may be throttling")
else:
    print("   ✗ VERY SLOW - Likely not using GPU properly")
    sys.exit(1)

# Test 4: Batch detection
print("\n4. Testing batch detection (32 frames)...")
batch_frames = [test_frame for _ in range(32)]

# Warmup
_ = model.predict(batch_frames, verbose=False, device='0')

# Timed run
start = time.time()
results = model.predict(batch_frames, verbose=False, conf=0.25, device='0')
end = time.time()

batch_time = (end - start) * 1000
per_frame = batch_time / 32

print(f"   Batch time: {batch_time:.0f}ms")
print(f"   Per frame: {per_frame:.1f}ms")

if batch_time < 2000:
    print("   ✓ EXCELLENT - Batch processing working!")
elif batch_time < 5000:
    print("   ⚠ ACCEPTABLE - Could be faster")
else:
    print(f"   ✗ TOO SLOW - {batch_time/1000:.1f}s for 32 frames")

# Test 5: GPU utilization check
print("\n5. GPU Memory Check...")
if torch.cuda.is_available():
    print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")
    print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.0f} MB")
    print(f"   Max allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.0f} MB")

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

if avg_time < 100 and batch_time < 3000:
    print("✓ GPU is properly configured and being used")
    print(f"  Expected processing speed: ~{32000 / batch_time:.1f} FPS on batches")
else:
    print("⚠ GPU is detected but performance is suboptimal")
    print(f"  Current: {per_frame:.1f}ms per frame")
    print(f"  Expected: <50ms per frame on GPU")

print()
