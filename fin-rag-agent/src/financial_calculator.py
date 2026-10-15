"""Extract annual statement figures and calculate year-over-year changes."""

import re
from decimal import Decimal


METRICS = {
    "revenue": {
        "labels": ("total net sales", "total revenue", "total revenues"),
        "query_label": "total net sales revenue",
        "end_markers": ("cost of sales:",),
    },
    "net income": {
        "labels": ("net income",),
        "query_label": "net income",
        "end_markers": ("earnings per share", "basic", "diluted"),
    },
    "operating income": {
        "labels": ("operating income",),
        "query_label": "operating income",
        "end_markers": ("other income", "income before"),
    },
    "gross margin": {
        "labels": ("gross margin", "gross profit"),
        "query_label": "gross margin gross profit",
        "end_markers": ("operating expenses:",),
    },
}


def parse_calculation_request(question: str) -> dict:
    """Parse the bounded, supported same-company annual change question."""
    years = sorted({int(year) for year in re.findall(r"\b20\d{2}\b", question)})
    if len(years) != 2:
        raise ValueError("Please specify exactly two fiscal years.")

    lowered = question.lower()
    companies = [name for name in ("apple", "tesla") if re.search(rf"\b{name}\b", lowered)]
    if len(companies) != 1:
        raise ValueError("Please specify Apple or Tesla as the company.")

    if "net income" in lowered or "profit" in lowered:
        metric = "net income"
    elif "operating income" in lowered:
        metric = "operating income"
    elif "gross margin" in lowered or "gross profit" in lowered:
        metric = "gross margin"
    elif any(term in lowered for term in ("revenue", "net sales", "sales")):
        metric = "revenue"
    else:
        raise ValueError("Supported metrics are revenue, net income, operating income, and gross margin.")

    difference_terms = ("absolute difference", "dollar difference", "difference in dollars")
    operation = "difference" if any(term in lowered for term in difference_terms) else "percentage_change"

    return {
        "company": companies[0],
        "metric": metric,
        "start_year": years[0],
        "end_year": years[1],
        "operation": operation,
    }


def build_retrieval_query(request: dict) -> str:
    metric = METRICS[request["metric"]]["query_label"]
    years = request["start_year"], request["end_year"]
    return (
        f"{request['company']} consolidated statements of operations "
        f"{metric} in millions years ended {years[0]} {years[1]}"
    )


def calculate_from_chunks(request: dict, chunks: list[dict]) -> dict:
    metric = METRICS[request["metric"]]
    for chunk in chunks:
        text = chunk["text"]
        lowered = text.lower()
        statement_start = lowered.find("consolidated statements of operations")
        if statement_start < 0:
            continue

        for label in sorted(metric["labels"], key=len, reverse=True):
            match = re.search(rf"\b{re.escape(label)}\b", lowered[statement_start:])
            if not match:
                continue
            row_start = statement_start + match.end()
            header_years = list(dict.fromkeys(
                int(year) for year in re.findall(r"\b20\d{2}\b", text[statement_start:statement_start + match.start()])
            ))
            if request["start_year"] not in header_years or request["end_year"] not in header_years:
                continue

            row_end = min(
                (position for marker in metric["end_markers"]
                 if (position := lowered.find(marker, row_start)) >= 0),
                default=min(len(text), row_start + 100),
            )
            row = text[row_start:row_end]
            numbers = []
            for number_match in re.finditer(r"\(?\d[\d,]*(?:\.\d+)?\)?", row):
                if re.match(r"\s*%", row[number_match.end():]):
                    continue
                value = number_match.group().replace(",", "")
                if value.startswith("(") and value.endswith(")"):
                    value = f"-{value[1:-1]}"
                numbers.append(Decimal(value))

            if len(numbers) < len(header_years):
                continue
            values_by_year = dict(zip(header_years, numbers))
            start_value = values_by_year[request["start_year"]]
            end_value = values_by_year[request["end_year"]]
            change = end_value - start_value
            percent_change = (change / start_value * 100) if start_value else None
            unit = "million USD" if re.search(r"in millions", lowered) else "USD"
            source = chunk.get("meta", {})
            return {
                "company": request["company"],
                "metric": request["metric"],
                "start_year": request["start_year"],
                "end_year": request["end_year"],
                "start_value": start_value,
                "end_value": end_value,
                "change": change,
                "percent_change": percent_change,
                "operation": request["operation"],
                "unit": unit,
                "sources": [{
                    "company": source.get("company", request["company"]),
                    "page": source.get("page"),
                    "source_file": source.get("source_file"),
                }],
                "evidence": text,
            }

    raise ValueError("Could not find both years for that metric in a retrieved statement table.")


def format_calculation(result: dict) -> str:
    company = result["company"].title()
    metric = result["metric"]
    start_year = result["start_year"]
    end_year = result["end_year"]
    start_value = result["start_value"]
    end_value = result["end_value"]
    source = result["sources"][0]
    citation = f"[{source['company'].title()}, p. {source['page']}]"

    if result["operation"] == "difference":
        return (
            f"{company} {metric} changed by {result['change']:,.0f} {result['unit']} "
            f"from {start_year} ({start_value:,.0f}) to {end_year} ({end_value:,.0f}). {citation}"
        )
    if result["percent_change"] is None:
        return "Percentage change is undefined because the starting value is zero."
    direction = "increased" if result["change"] > 0 else "decreased" if result["change"] < 0 else "was unchanged"
    return (
        f"{company} {metric} {direction} by {abs(result['percent_change']):.2f}% "
        f"from {start_year} ({start_value:,.0f} {result['unit']}) "
        f"to {end_year} ({end_value:,.0f} {result['unit']}). {citation}"
    )