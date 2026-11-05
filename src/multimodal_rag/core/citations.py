import re


def validate_citations(answer: str, source_count: int) -> dict:
    """
    Check whether an answer contains citations and whether
    their source numbers exist in the current source list.

    Note:
    This validates citation labels, not whether the cited
    passages semantically support every claim.
    """

    citations = [
        int(number)
        for number in re.findall(
            r"\[Source\s+(\d+)\]",
            answer,
            flags=re.IGNORECASE
        )
    ]

    if not citations:
        return {
            "valid": False,
            "reason": "Missing citations",
            "citations": [],
            "invalid": []
        }

    invalid = sorted({
        number
        for number in citations
        if number < 1 or number > source_count
    })

    if invalid:
        return {
            "valid": False,
            "reason": "Invalid source numbers",
            "citations": citations,
            "invalid": invalid
        }

    return {
        "valid": True,
        "reason": "Citations exist and reference available sources",
        "citations": citations,
        "invalid": []
    }
