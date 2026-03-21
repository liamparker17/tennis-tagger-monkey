#!/usr/bin/env python3
"""
Demo Script - Tennis Tagger

Demonstrates the complete tennis tagging workflow with sample data.

Usage:
    python scripts/demo.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import TennisTagger


def create_sample_video(output_path: str = "demo_match.mp4", duration: int = 30):
    """
    Create a sample tennis video for demonstration.
    
    Args:
        output_path: Path for output video
        duration: Duration in seconds
    """
    print(f"📹 Creating sample video ({duration}s)...")
    
    fps = 30
    width, height = 1280, 720
    total_frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        # Create frame with tennis court-like background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        # Draw court lines (simplified)
        green = (100, 180, 100)
        white = (255, 255, 255)
        
        # Fill court area
        cv2.rectangle(frame, (100, 100), (width-100, height-100), green, -1)
        
        # Draw court lines
        cv2.rectangle(frame, (100, 100), (width-100, height-100), white, 3)
        cv2.line(frame, (width//2, 100), (width//2, height-100), white, 2)
        
        # Add moving "players" (circles)
        player1_y = int(height * 0.7 + 50 * np.sin(frame_num * 0.1))
        player2_y = int(height * 0.3 + 50 * np.sin(frame_num * 0.1 + np.pi))
        
        cv2.circle(frame, (300, player1_y), 30, (255, 0, 0), -1)
        cv2.circle(frame, (width-300, player2_y), 30, (0, 0, 255), -1)
        
        # Add moving "ball"
        ball_x = int(width/2 + 400 * np.sin(frame_num * 0.15))
        ball_y = int(height/2 + 200 * np.sin(frame_num * 0.3))
        cv2.circle(frame, (ball_x, ball_y), 8, (0, 255, 255), -1)
        
        # Add text
        text = f"Demo Match - Frame {frame_num}/{total_frames}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Sample video created: {output_path}")


def run_demo():
    """Run complete demonstration workflow."""
    print("\n" + "="*60)
    print("🎾 Tennis Tagger Demo")
    print("="*60 + "\n")
    
    # Step 1: Create sample video
    video_path = "demo_match.mp4"
    if not os.path.exists(video_path):
        create_sample_video(video_path, duration=30)
    else:
        print(f"📹 Using existing video: {video_path}")
    
    # Step 2: Initialize tagger
    print("\n📦 Initializing Tennis Tagger...")
    try:
        tagger = TennisTagger(device="auto")
    except Exception as e:
        print(f"❌ Error initializing tagger: {e}")
        print("\nNote: This demo requires all dependencies installed.")
        print("Run: pip install -r requirements.txt")
        return
    
    # Step 3: Process video
    print("\n🎬 Processing video...")
    output_csv = "demo_tags.csv"
    
    try:
        stats = tagger.process_video(
            video_path=video_path,
            output_path=output_csv
        )
        
        print(f"\n✅ Processing complete!")
        print(f"\n📊 Results:")
        print(f"   - CSV saved: {output_csv}")
        print(f"   - Processing time: {stats['processing_time']:.1f}s")
        print(f"   - Speed factor: {stats['speed_factor']:.1f}x")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print("\nNote: This is a simplified demo with synthetic data.")
        return
    
    # Step 4: Display results
    print(f"\n📄 CSV Preview:")
    try:
        import pandas as pd
        df = pd.read_csv(output_csv)
        print(df.head())
        print(f"\nTotal rows: {len(df)}")
    except Exception as e:
        print(f"Could not display CSV: {e}")
    
    print("\n" + "="*60)
    print("✨ Demo Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Open demo_tags.csv in Excel to view full results")
    print("2. Try with your own tennis video:")
    print("   python main.py --video your_match.mp4 --output tags.csv")
    print("3. Launch GUI for interactive mode:")
    print("   python gui/app.py")
    print("="*60 + "\n")


def main():
    """Main demo entry point."""
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        print("\nFor help, see QUICKSTART.md")


if __name__ == "__main__":
    main()
