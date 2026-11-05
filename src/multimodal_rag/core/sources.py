from urllib.parse import urlparse


def get_source_info(doc):
    metadata = doc.metadata

    page = metadata.get("page")
    page_label = metadata.get("page_label")

    if page_label is not None:
        page_reference = str(page_label)
    elif page is not None:
        page_reference = str(int(page) + 1)
    else:
        page_reference = None

    raw_source = str(
        metadata.get("source", "Unknown source")
    )

    title = (
        metadata.get("title")
        or metadata.get("video_title")
        or metadata.get("document_name")
    )

    video_id = metadata.get("video_id")
    start_seconds = metadata.get("start_seconds")

    url = None
    timestamp_url = None

    # YouTube source
    if video_id:
        url = (
            "https://www.youtube.com/watch?v="
            f"{video_id}"
        )

        if start_seconds is not None:
            try:
                seconds = max(0, int(float(start_seconds)))
                timestamp_url = f"{url}&t={seconds}s"
            except (TypeError, ValueError):
                pass

        display_name = title or f"YouTube video ({video_id})"

    # Other web source
    elif raw_source.startswith(("https://", "http://")):
        url = raw_source
        parsed_url = urlparse(raw_source)

        display_name = (
            title
            or parsed_url.path.rstrip("/").split("/")[-1]
            or parsed_url.netloc
        )

    # Local file or document
    else:
        display_name = (
            title
            or raw_source.replace("\\", "/").rstrip("/").split("/")[-1]
            or "Unknown source"
        )

    return {
        "name": display_name,
        "raw_source": raw_source,
        "url": url,
        "timestamp_url": timestamp_url,
        "language": metadata.get("language", "Unknown"),
        "start_seconds": start_seconds,
        "page_reference": page_reference,
    }
