"""Example: Airtable Integration - Async Version

This example demonstrates how to use SceneFlow's async API with Airtable integration.

Note: Currently, the Airtable integration functions (analyze_and_upload_to_airtable, etc.)
are synchronous. This example shows how to use them in an async context using asyncio.to_thread()
to avoid blocking the event loop.

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

import asyncio
import os
from sceneflow.integration import (
    analyze_and_upload_to_airtable,
    analyze_ranked_and_upload_to_airtable,
    cut_and_upload_to_airtable,
)


async def example_analyze_and_upload():
    """Example 1: Basic analysis with Airtable upload (async wrapper)"""
    print("Example 1: Analyze and upload to Airtable (Async)")
    print("=" * 60)

    video_path = "path/to/your/video.mp4"

    # Set credentials (or use environment variables)
    os.environ["AIRTABLE_ACCESS_TOKEN"] = "your-access-token-here"
    os.environ["AIRTABLE_BASE_ID"] = "appXXXXXXXXXXXXXX"

    try:
        # Run the sync function in a thread pool to avoid blocking the event loop
        timestamp, record_id = await asyncio.to_thread(
            analyze_and_upload_to_airtable,
            video_path,
            sample_rate=2,  # Process every 2nd frame for speed
        )
        print(f"✓ Cut at: {timestamp:.2f}s")
        print(f"✓ Uploaded to Airtable: {record_id}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print()


async def example_ranked_and_upload():
    """Example 2: Get top 5 cut points and upload best result (async wrapper)"""
    print("Example 2: Top 5 cut points with Airtable upload (Async)")
    print("=" * 60)

    video_path = "path/to/your/video.mp4"

    try:
        # Run the sync function in a thread pool
        timestamps, record_id = await asyncio.to_thread(
            analyze_ranked_and_upload_to_airtable,
            video_path,
            n=5,
            sample_rate=2,
        )
        print(f"✓ Top 5 cut points: {[f'{t:.2f}s' for t in timestamps]}")
        print(f"✓ Best result uploaded to Airtable: {record_id}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print()


async def example_cut_and_upload():
    """Example 3: Cut video and upload to Airtable (async wrapper)"""
    print("Example 3: Cut video and upload to Airtable (Async)")
    print("=" * 60)

    video_path = "path/to/your/video.mp4"
    output_video = "output/cut_video_async.mp4"

    try:
        # Run the sync function in a thread pool
        timestamp, record_id = await asyncio.to_thread(
            cut_and_upload_to_airtable,
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


async def example_explicit_credentials():
    """Example 4: Using explicit credentials (async wrapper)"""
    print("Example 4: Explicit credentials (Async)")
    print("=" * 60)

    video_path = "path/to/your/video.mp4"

    try:
        # Run the sync function in a thread pool
        timestamp, record_id = await asyncio.to_thread(
            analyze_and_upload_to_airtable,
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


async def example_with_llm():
    """Example 5: With LLM selection (async wrapper)"""
    print("Example 5: LLM selection + Airtable upload (Async)")
    print("=" * 60)

    video_path = "path/to/your/video.mp4"
    os.environ["OPENAI_API_KEY"] = "your-openai-key-here"

    try:
        # Run the sync function in a thread pool
        timestamp, record_id = await asyncio.to_thread(
            analyze_and_upload_to_airtable,
            video_path,
            use_llm_selection=True,  # Use GPT-4o vision for final selection
            sample_rate=2,
        )
        print(f"✓ LLM selected cut at: {timestamp:.2f}s")
        print(f"✓ Uploaded to Airtable: {record_id}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print()


async def main():
    print("SceneFlow - Airtable Integration Async Examples")
    print("=" * 60)
    print("\nNote: These examples use asyncio.to_thread() to run")
    print("synchronous Airtable functions without blocking the event loop.")
    print("\nUncomment the examples you want to run.\n")

    # Uncomment the examples you want to run:

    # await example_analyze_and_upload()
    # await example_ranked_and_upload()
    # await example_cut_and_upload()
    # await example_explicit_credentials()
    # await example_with_llm()

    print("=" * 60)
    print("Note: Check your Airtable base to see the uploaded results!")
    print("The table will contain:")
    print("  - Original video file")
    print("  - Selected frame image")
    print("  - Cut video from start to cut point")
    print("  - Complete JSON analysis data")


if __name__ == "__main__":
    asyncio.run(main())
