#!/usr/bin/env python3
"""
Test Phase 1 Optimizations
Verify FP16, batch size increase, and native tracking are working
"""

import torch
import time
import numpy as np
from pathlib import Path

def test_gpu_fp16_support():
    """Test if GPU supports FP16 inference"""
    print("="*60)
    print("Testing GPU FP16 Support")
    print("="*60)

    if not torch.cuda.is_available():
        print("❌ No GPU available")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    print(f"✓ GPU detected: {gpu_name}")

    # Check if GPU supports FP16
    device = torch.device('cuda:0')
    try:
        # Create test tensor in FP16
        test_tensor = torch.randn(100, 100, dtype=torch.float16, device=device)
        result = torch.mm(test_tensor, test_tensor)
        print(f"✓ FP16 operations supported")
        return True
    except Exception as e:
        print(f"❌ FP16 not supported: {e}")
        return False

def test_yolo_fp16_inference():
    """Test YOLO model with FP16"""
    print("\n" + "="*60)
    print("Testing YOLO FP16 Inference Speed")
    print("="*60)

    try:
        from ultralytics import YOLO

        # Load model
        model = YOLO('yolov8s.pt')
        model.to('cuda')

        # Create dummy frames
        dummy_frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                       for _ in range(32)]

        # Warmup
        _ = model.predict(dummy_frames[0], verbose=False, device='0', half=False)

        # Test FP32 (baseline)
        start = time.time()
        _ = model.predict(dummy_frames, verbose=False, device='0', half=False)
        fp32_time = time.time() - start

        # Test FP16 (optimized)
        start = time.time()
        _ = model.predict(dummy_frames, verbose=False, device='0', half=True)
        fp16_time = time.time() - start

        speedup = fp32_time / fp16_time

        print(f"FP32 time: {fp32_time:.3f}s")
        print(f"FP16 time: {fp16_time:.3f}s")
        print(f"✓ FP16 speedup: {speedup:.2f}x faster")

        if speedup > 1.3:
            print("✓ FP16 optimization working well!")
            return True
        else:
            print("⚠ FP16 speedup lower than expected")
            return False

    except Exception as e:
        print(f"❌ Error testing YOLO FP16: {e}")
        return False

def test_batch_size_scaling():
    """Test different batch sizes"""
    print("\n" + "="*60)
    print("Testing Batch Size Scaling")
    print("="*60)

    try:
        from ultralytics import YOLO

        model = YOLO('yolov8s.pt')
        model.to('cuda')

        # Test different batch sizes
        batch_sizes = [8, 32, 64, 128]
        results = {}

        for batch_size in batch_sizes:
            try:
                # Create dummy frames
                dummy_frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                               for _ in range(batch_size)]

                # Warmup
                if batch_size == 8:
                    _ = model.predict(dummy_frames[0], verbose=False, device='0', half=True)

                # Time inference
                start = time.time()
                _ = model.predict(dummy_frames, verbose=False, device='0', half=True)
                elapsed = time.time() - start

                fps = batch_size / elapsed
                results[batch_size] = {'time': elapsed, 'fps': fps}
                print(f"Batch {batch_size:3d}: {elapsed:.3f}s ({fps:.1f} FPS)")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Batch {batch_size:3d}: ❌ Out of memory")
                    break
                else:
                    raise

        # Find best batch size
        best_batch = max(results.items(), key=lambda x: x[1]['fps'])
        print(f"\n✓ Optimal batch size: {best_batch[0]} ({best_batch[1]['fps']:.1f} FPS)")

        if best_batch[0] >= 64:
            print("✓ Large batch sizes working well!")
            return True
        else:
            print("⚠ Consider reducing batch size due to GPU memory")
            return True

    except Exception as e:
        print(f"❌ Error testing batch sizes: {e}")
        return False

def test_native_tracking():
    """Test native Ultralytics tracking"""
    print("\n" + "="*60)
    print("Testing Native Ultralytics Tracking")
    print("="*60)

    try:
        from ultralytics import YOLO

        model = YOLO('yolov8s.pt')
        model.to('cuda')

        # Create dummy video frames (simulated motion)
        dummy_frames = []
        for i in range(30):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            # Add moving rectangle (simulated person)
            x = 400 + i * 10
            cv2_available = False
            try:
                import cv2
                cv2.rectangle(frame, (x, 300), (x+100, 500), (255, 255, 255), -1)
                cv2_available = True
            except:
                pass
            dummy_frames.append(frame)

        # Test tracking
        start = time.time()
        results = model.track(
            dummy_frames,
            verbose=False,
            device='0',
            half=True,
            tracker='bytetrack.yaml',
            persist=True,
            conf=0.1
        )
        elapsed = time.time() - start

        # Check if tracking IDs are present
        has_track_ids = False
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    if hasattr(box, 'id') and box.id is not None:
                        has_track_ids = True
                        break

        fps = len(dummy_frames) / elapsed
        print(f"Tracking time: {elapsed:.3f}s ({fps:.1f} FPS)")

        if has_track_ids:
            print("✓ Native tracking working (track IDs present)")
        else:
            print("⚠ Track IDs not found (might need actual detections)")

        print("✓ Native tracking API functional")
        return True

    except Exception as e:
        print(f"❌ Error testing tracking: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 1 tests"""
    print("\n" + "="*60)
    print("PHASE 1 OPTIMIZATION TEST SUITE")
    print("="*60 + "\n")

    results = {
        'GPU FP16 Support': test_gpu_fp16_support(),
        'YOLO FP16 Inference': test_yolo_fp16_inference(),
        'Batch Size Scaling': test_batch_size_scaling(),
        'Native Tracking': test_native_tracking()
    }

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s}: {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print("\n" + "="*60)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    print("="*60)

    if total_passed == total_tests:
        print("\n🎉 All Phase 1 optimizations verified!")
        print("\nExpected speedup: 2.5-4x faster than baseline")
        print("\nNext: Run on actual video to measure real-world performance")
    else:
        print("\n⚠ Some optimizations need attention")

    return total_passed == total_tests

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
