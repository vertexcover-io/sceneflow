"""External integrations package.

This package provides modules for integrating with external services
like Airtable.
"""

from sceneflow.integration.airtable import AirtableUploader, upload_to_airtable
from sceneflow.integration.airtable_api import (
    analyze_and_upload_to_airtable,
    analyze_ranked_and_upload_to_airtable,
    cut_and_upload_to_airtable,
)

__all__ = [
    # Low-level Airtable utilities
    "AirtableUploader",
    "upload_to_airtable",
    # High-level Airtable integration APIs
    "analyze_and_upload_to_airtable",
    "analyze_ranked_and_upload_to_airtable",
    "cut_and_upload_to_airtable",
]
