import os


GOOGLE_DOC_SCOPES = ["https://www.googleapis.com/auth/documents"]


def export_text_to_google_doc(text, title="Tailored CV Draft", service_account_file=None):
    """
    Create a Google Doc and populate it with the supplied text.

    Requires:
    - google-api-python-client
    - google-auth
    - A service account JSON key with Docs API access.
    """
    key_path = service_account_file or os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
    if not key_path:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_FILE is not configured.")

    try:
        from google.oauth2 import service_account  # pyright: ignore[reportMissingImports]
        from googleapiclient.discovery import build  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Google Docs dependencies missing. Install: pip install google-api-python-client google-auth"
        ) from exc

    credentials = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=GOOGLE_DOC_SCOPES,
    )
    docs_service = build("docs", "v1", credentials=credentials)

    document = docs_service.documents().create(body={"title": title}).execute()
    document_id = document["documentId"]
    docs_service.documents().batchUpdate(
        documentId=document_id,
        body={
            "requests": [
                {
                    "insertText": {
                        "location": {"index": 1},
                        "text": text,
                    }
                }
            ]
        },
    ).execute()

    return f"https://docs.google.com/document/d/{document_id}/edit"
