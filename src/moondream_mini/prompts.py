from __future__ import annotations


def normalize_text(s: str) -> str:
    return " ".join(s.strip().split())


def infer_label_space(question: str) -> tuple[str, str]:
    q = normalize_text(question).lower()
    if q.startswith(("how many", "how much", "count")):
        return "count", "count_4"
    if q.startswith(("where", "which side")):
        return "location", "location_3"
    return "yes_no", "yes_no"


def resolve_label_space(task_type: str | None, label_space: str | None, question: str) -> tuple[str, str]:
    inferred_task_type, inferred_label_space = infer_label_space(question)
    normalized_task_type = normalize_text(task_type).lower() if task_type else ""
    if normalized_task_type in {"yes_no", "count", "location"}:
        task_type = normalized_task_type
        if not label_space:
            label_space = {"yes_no": "yes_no", "count": "count_4", "location": "location_3"}[task_type]
    else:
        task_type = inferred_task_type
    if not label_space:
        label_space = inferred_label_space
    if task_type == "location" and label_space == "location_9":
        label_space = "location_3"
    return task_type, label_space


def build_prompt(question: str, label_space: str | None = None) -> str:
    q = normalize_text(question)
    space = label_space or infer_label_space(q)[1]
    if space == "count_4":
        options = "0, 1, 2, 3+"
    elif space == "location_3":
        options = "left, center, right"
    elif space == "location_9":
        options = (
            "top-left, top-center, top-right, center-left, center, "
            "center-right, bottom-left, bottom-center, bottom-right"
        )
    else:
        options = "yes, no"
    return f"<image> Question: {q} Options: {options} Answer:"


def extract_answer(text: str, label_space: str) -> str:
    norm = normalize_text(text)
    if "Answer:" in norm:
        norm = norm.split("Answer:", 1)[1].strip()
    for stop in ("<eos>", "</s>"):
        if stop in norm:
            norm = norm.split(stop, 1)[0].strip()
    norm = norm.lower()
    if label_space == "count_4":
        if norm.startswith("3"):
            return "3+"
        for opt in ("2", "1", "0"):
            if norm.startswith(opt):
                return opt
    if label_space == "location_3":
        for opt in ("left", "center", "right"):
            if norm.startswith(opt) or opt in norm:
                return opt
    if label_space == "yes_no":
        if norm.startswith("y"):
            return "yes"
        if norm.startswith("n"):
            return "no"
    return norm.split()[0] if norm else norm
