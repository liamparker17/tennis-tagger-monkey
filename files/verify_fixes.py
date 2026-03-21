"""
Verification script to check all new methods exist.
Run this to verify all AttributeError issues are fixed.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 70)
print("VERIFICATION: Checking all new methods exist")
print("=" * 70)
print()

errors = []
warnings = []

# Check 1: Utility modules
print("1. Checking utility modules...")
try:
    from utils.checkpointing import save_checkpoint, load_checkpoint, clear_checkpoint
    print("   ✓ utils.checkpointing imports OK")
except Exception as e:
    errors.append(f"utils.checkpointing: {e}")
    print(f"   ✗ utils.checkpointing FAILED: {e}")

try:
    from utils.stability import SceneStabilityChecker
    print("   ✓ utils.stability imports OK")
except Exception as e:
    errors.append(f"utils.stability: {e}")
    print(f"   ✗ utils.stability FAILED: {e}")

print()

# Check 2: VideoProcessor methods
print("2. Checking VideoProcessor methods...")
try:
    from video_processor import VideoProcessor
    vp = VideoProcessor({'fps': 30})

    methods_to_check = ['get_video_metadata', 'iter_frames', 'iter_frame_chunks']
    for method in methods_to_check:
        if hasattr(vp, method):
            print(f"   ✓ VideoProcessor.{method} exists")
        else:
            errors.append(f"VideoProcessor.{method} missing")
            print(f"   ✗ VideoProcessor.{method} MISSING")

except Exception as e:
    errors.append(f"VideoProcessor import: {e}")
    print(f"   ✗ VideoProcessor import FAILED: {e}")

print()

# Check 3: PlayerDetector methods
print("3. Checking PlayerDetector methods...")
try:
    from detection.player_detector import PlayerDetector

    # Check class has the method (don't instantiate to avoid model loading)
    if hasattr(PlayerDetector, 'detect'):
        print("   ✓ PlayerDetector.detect exists")
    else:
        errors.append("PlayerDetector.detect missing")
        print("   ✗ PlayerDetector.detect MISSING")

    if hasattr(PlayerDetector, 'detect_batch'):
        print("   ✓ PlayerDetector.detect_batch exists")
    else:
        errors.append("PlayerDetector.detect_batch missing")
        print("   ✗ PlayerDetector.detect_batch MISSING")

    if hasattr(PlayerDetector, 'track'):
        print("   ✓ PlayerDetector.track exists")
    else:
        errors.append("PlayerDetector.track missing")
        print("   ✗ PlayerDetector.track MISSING")

except Exception as e:
    errors.append(f"PlayerDetector import: {e}")
    print(f"   ✗ PlayerDetector import FAILED: {e}")

print()

# Check 4: BallDetector methods
print("4. Checking BallDetector methods...")
try:
    from detection.ball_detector import BallDetector

    if hasattr(BallDetector, 'detect'):
        print("   ✓ BallDetector.detect exists")
    else:
        errors.append("BallDetector.detect missing")
        print("   ✗ BallDetector.detect MISSING")

    if hasattr(BallDetector, 'detect_batch'):
        print("   ✓ BallDetector.detect_batch exists")
    else:
        errors.append("BallDetector.detect_batch missing")
        print("   ✗ BallDetector.detect_batch MISSING")

    if hasattr(BallDetector, 'track'):
        print("   ✓ BallDetector.track exists")
    else:
        errors.append("BallDetector.track missing")
        print("   ✗ BallDetector.track MISSING")

except Exception as e:
    errors.append(f"BallDetector import: {e}")
    print(f"   ✗ BallDetector import FAILED: {e}")

print()

# Check 5: video_processing_thread updated
print("5. Checking video_processing_thread...")
try:
    with open('video_processing_thread.py', 'r') as f:
        content = f.read()
        if 'batch_size=' in content and 'checkpoint_interval=' in content:
            print("   ✓ video_processing_thread.py has new parameters")
        else:
            warnings.append("video_processing_thread.py may not have new parameters")
            print("   ⚠ video_processing_thread.py missing new parameters")
except Exception as e:
    warnings.append(f"Could not check video_processing_thread.py: {e}")
    print(f"   ⚠ Could not verify video_processing_thread.py: {e}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

if errors:
    print(f"\n❌ {len(errors)} ERRORS FOUND:")
    for error in errors:
        print(f"   - {error}")
    print("\nSome methods are still missing. Check the error messages above.")
    sys.exit(1)
elif warnings:
    print(f"\n⚠ {len(warnings)} WARNINGS:")
    for warning in warnings:
        print(f"   - {warning}")
    print("\nAll critical methods exist, but there are warnings.")
    print("The code should work but may have issues.")
    sys.exit(0)
else:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nAll new methods exist. AttributeError issues should be resolved.")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Test with: python main.py --video test.mp4 --output test.csv")
    sys.exit(0)
