"""Example: Airtable Integration

This example demonstrates how to use SceneFlow's Airtable integration
to automatically upload analysis results to your Airtable base.

Prerequisites:
    1. Install pyairtable: uv add pyairtable
    2. Set environment variables:
       - AIRTABLE_ACCESS_TOKEN: Your Airtable personal access token
       - AIRTABLE_BASE_ID: Your Airtable base ID (e.g., appXXXXXXXXXXXXXX)
       - AIRTABLE_TABLE_NAME: Optional table name (defaults to "SceneFlow Analysis")

The Airtable integration creates a table with these fields:
    - Timestamp: Cut point timestamp
    - Raw Video: Original video file (attachment)
    - Selected Frame Image: Best frame as JPEG (attachment)
    - Output Video: Cut video from start to cut point (attachment)
    - Raw Data: Complete JSON analysis details
"""

import os
from sceneflow.integration import (
    analyze_and_upload_to_airtable,
    analyze_ranked_and_upload_to_airtable,
    cut_and_upload_to_airtable,
)

# Example 1: Basic analysis with Airtable upload
print("Example 1: Analyze and upload to Airtable")
print("=" * 60)

video_path = "path/to/your/video.mp4"

# Set credentials (or use environment variables)
os.environ["AIRTABLE_ACCESS_TOKEN"] = "your-access-token-here"
os.environ["AIRTABLE_BASE_ID"] = "appXXXXXXXXXXXXXX"

try:
    timestamp, record_id = analyze_and_upload_to_airtable(
        video_path,
        sample_rate=2,  # Process every 2nd frame for speed
    )
    print(f"✓ Cut at: {timestamp:.2f}s")
    print(f"✓ Uploaded to Airtable: {record_id}")
except Exception as e:
    print(f"✗ Error: {e}")

print()

# Example 2: Get top 5 cut points and upload best result
print("Example 2: Top 5 cut points with Airtable upload")
print("=" * 60)

try:
    timestamps, record_id = analyze_ranked_and_upload_to_airtable(
        video_path,
        n=5,
        sample_rate=2,
    )
    print(f"✓ Top 5 cut points: {[f'{t:.2f}s' for t in timestamps]}")
    print(f"✓ Best result uploaded to Airtable: {record_id}")
except Exception as e:
    print(f"✗ Error: {e}")

print()

# Example 3: Cut video and upload to Airtable
print("Example 3: Cut video and upload to Airtable")
print("=" * 60)

output_video = "output/cut_video.mp4"

try:
    timestamp, record_id = cut_and_upload_to_airtable(
        video_path,
        output_path=output_video,
        save_frames=True,  # Also save annotated frames
        sample_rate=2,
    )
    print(f"✓ Cut video saved: {output_video}")
    print(f"✓ Cut at: {timestamp:.2f}s")
    print(f"✓ Uploaded to Airtable: {record_id}")
except Exception as e:
    print(f"✗ Error: {e}")

print()

# Example 4: Using explicit credentials (no environment variables)
print("Example 4: Explicit credentials")
print("=" * 60)

try:
    timestamp, record_id = analyze_and_upload_to_airtable(
        video_path,
        airtable_access_token="your-token",
        airtable_base_id="appXXXXXXXXXXXXXX",
        airtable_table_name="My Custom Table",
        sample_rate=2,
    )
    print(f"✓ Cut at: {timestamp:.2f}s")
    print(f"✓ Uploaded to custom table: {record_id}")
except Exception as e:
    print(f"✗ Error: {e}")

print()

# Example 5: With LLM selection
print("Example 5: LLM selection + Airtable upload")
print("=" * 60)

os.environ["OPENAI_API_KEY"] = "your-openai-key-here"

try:
    timestamp, record_id = analyze_and_upload_to_airtable(
        video_path,
        use_llm_selection=True,  # Use GPT-4o vision for final selection
        sample_rate=2,
    )
    print(f"✓ LLM selected cut at: {timestamp:.2f}s")
    print(f"✓ Uploaded to Airtable: {record_id}")
except Exception as e:
    print(f"✗ Error: {e}")

print()
print("=" * 60)
print("Note: Check your Airtable base to see the uploaded results!")
print("The table will contain:")
print("  - Original video file")
print("  - Selected frame image")
print("  - Cut video from start to cut point")
print("  - Complete JSON analysis data")
