"""Droplet pack: what a person can review and correct.

Droplets are listed one per row of the droplet table (flagged ones first); outlines are
edited on the segmentation labels; excluding a droplet removes its whole track; the cell
count and the positive/negative call can be set by hand.
"""

from __future__ import annotations

from typing import Any

from vexel.engine.recipe import Recipe
from vexel.engine.review import EditableField, ReviewAdapter

FLAG_TEXT = {
    "low_circularity": "形状不圆",
    "size_outlier": "粒径异常",
    "count_unstable": "计数不稳",
    "near_threshold": "近阈值",
    "few_frames": "帧数少",
}


class DropletAdapter(ReviewAdapter):
    def describe(self, row: dict[str, Any]) -> str:
        parts = []
        if row.get("diameter") is not None:
            parts.append(f"直径 {row['diameter']:.3g}")
        if row.get("n_inclusions") is not None and row.get("n_inclusions") == row.get("n_inclusions"):
            parts.append(f"细胞 {int(row['n_inclusions'])}")
        if "positive" in row:
            parts.append("阳性" if str(row["positive"]).lower() in ("true", "1") else "阴性")
        return " · ".join(parts)


def adapter(recipe: Recipe) -> ReviewAdapter | None:
    steps = {s.op: s.id for s in recipe.steps}
    if "droplet.summary" not in steps or "track.link" not in steps:
        return None
    summary, track = steps["droplet.summary"], steps["track.link"]
    fields = []
    if "droplet.inclusions" in steps:
        fields.append(EditableField("n_inclusions", "细胞数", f"{summary}.droplets", "int",
                                    help="该液滴里的细胞（或珠子）个数；改后泊松统计按此重算"))  # fmt: skip
    if "droplet.positivity" in steps:
        fields.append(EditableField("positive_manual", "阴阳性", f"{summary}.droplets", "bool", shown="positive",
                                    help="人工判定阳性或阴性；改后阳性比例与浓度按此重算"))  # fmt: skip
    objects = f"{steps['droplet.positivity']}.droplets" if "droplet.positivity" in steps else f"{summary}.droplets"
    flags = f"{steps['droplet.flags']}.table" if "droplet.flags" in steps else None
    labels = recipe.review.labels if recipe.review else "seg.labels"
    return DropletAdapter(objects=objects, labels=labels, exclude=f"{track}.table", flags=flags, fields=fields, object_name="液滴",
                          reasons=FLAG_TEXT)
