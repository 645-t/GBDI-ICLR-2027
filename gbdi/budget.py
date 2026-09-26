def render_budget_ledger(rendered: str, m: int) -> str:
    if not rendered:
        return rendered
    old = "Five independent inspections examined semantically equivalent views"
    if rendered.count(old) != 1:
        raise RuntimeError("Archived ledger preamble changed")
    new = {
        1: "One inspection examined a semantically equivalent view",
        3: "Three independent inspections examined semantically equivalent views",
    }[m]
    rendered = rendered.replace(old, new)
    if m == 1:
        old_consensus = "Consensus is an inspection-priority signal"
        if rendered.count(old_consensus) != 1:
            raise RuntimeError("Archived ledger nomination rule changed")
        rendered = rendered.replace(old_consensus, "A nomination is an inspection-priority signal")
    return rendered
