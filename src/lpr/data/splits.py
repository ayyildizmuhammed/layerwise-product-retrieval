"""Source-group-disjoint evaluation splits.

All split decisions are taken at the level of source groups (the shelf photograph a crop was cut
from), so every group belongs to exactly one role: gallery, validation-known, test-known,
validation-unknown or test-unknown.

1. Unknown products are drawn in a seeded random order: ``num_val_unknown`` validation-unknown
   SKUs, then ``num_test_unknown`` test-unknown SKUs, of which ``num_cross_family_test_unknown``
   are drawn as whole families (cross-family unknowns) and the rest from families that keep a
   catalogue SKU (same-family unknowns). A draw is accepted only if every other SKU keeps enough
   independent photographs: at least two unknown-free photographs for a catalogue SKU, so that it
   can have both gallery and query crops, and at least one test-unknown-free photograph for a
   validation-unknown SKU. The remaining SKUs form the catalogue.
2. A photograph that contains an unknown SKU goes to that unknown role (test-unknown wins when a
   photograph holds both). Only the unknown SKUs' crops are kept there: known crops that share a
   source image with an unknown SKU are removed rather than reassigned.
3. The remaining photographs contain catalogue SKUs only. For every catalogue SKU, photographs are
   drawn into the gallery until it holds ``gallery_fraction`` of the SKU's crops (at least one,
   at most ``max_gallery_per_sku``), always leaving one photograph for its queries; a photograph
   moves with all its crops. The other photographs become known queries, a ``val_fraction``
   share of them validation-known queries.
4. Every query that is byte-identical to a gallery image is removed.

A test-unknown SKU is a *same-family* unknown when another SKU of its family is in the catalogue
and a *cross-family* unknown otherwise (:mod:`lpr.evaluation.benchmark`).
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from lpr.data.manifest import Sample

ROLES = ("gallery", "val_known", "test_known", "val_unknown", "test_unknown")


@dataclass
class SplitConfig:
    """Split settings."""

    num_val_unknown: int = 10
    num_test_unknown: int = 91
    gallery_fraction: float = 0.3
    max_gallery_per_sku: int = 10
    val_fraction: float = 0.06
    num_cross_family_test_unknown: int | None = 63
    seed: int = 42


@dataclass
class SplitResult:
    """The five roles and what the split rules removed."""

    roles: dict[str, list[Sample]]
    removed_known_in_unknown_groups: int = 0
    removed_duplicates: int = 0
    unused_skus: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        """SKUs per role and the same-/cross-family composition of the test unknowns."""
        catalogue_families = {s.family_id for s in self.roles["gallery"]}
        test_unknown = {s.sku_id: s.family_id for s in self.roles["test_unknown"]}
        same = [sku for sku, family in test_unknown.items() if family in catalogue_families]
        return {
            "skus": {
                role: len({s.sku_id for s in samples}) for role, samples in self.roles.items()
            },
            "test_unknown_same_family_skus": len(same),
            "test_unknown_cross_family_skus": len(test_unknown) - len(same),
            "unused_skus": self.unused_skus,
        }


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _draw_unknowns(
    skus: list[str],
    groups_of: dict[str, set[str]],
    family_of: dict[str, str],
    config: SplitConfig,
    rng: random.Random,
) -> tuple[set[str], set[str]]:
    """Draw validation-unknown and test-unknown SKUs in random order, keeping the rest usable.

    Validation-unknown SKUs are drawn one by one. With ``num_cross_family_test_unknown`` set,
    that many test-unknown SKUs are drawn as whole families (no SKU of their family stays in the
    catalogue, so they are cross-family unknowns) and the rest one by one from families that keep
    a catalogue SKU (same-family unknowns).
    """
    val_unknown: set[str] = set()
    test_unknown: set[str] = set()
    state = {"blocked": set(), "test_blocked": set()}

    def try_add(new: set[str], to_val: bool) -> bool:
        blocked = state["blocked"] | set().union(*(groups_of[s] for s in new))
        test_blocked = state["test_blocked"] if to_val else (
            state["test_blocked"] | set().union(*(groups_of[s] for s in new))
        )
        unknown_now = val_unknown | test_unknown | new
        if any(len(groups_of[o] - blocked) < 2 for o in skus if o not in unknown_now):
            return False
        validation = val_unknown | (new if to_val else set())
        if any(len(groups_of[v] - test_blocked) < 1 for v in validation):
            return False
        (val_unknown if to_val else test_unknown).update(new)
        state["blocked"], state["test_blocked"] = blocked, test_blocked
        return True

    order = list(skus)
    rng.shuffle(order)
    for sku in order:
        if len(val_unknown) >= config.num_val_unknown:
            break
        try_add({sku}, to_val=True)

    members: dict[str, set[str]] = defaultdict(set)
    for sku in skus:
        members[family_of.get(sku, sku)].add(sku)

    cross_target = config.num_cross_family_test_unknown
    if cross_target:
        families = sorted(members)
        rng.shuffle(families)
        for family in families:
            if len(test_unknown) >= cross_target:
                break
            whole = members[family] - val_unknown
            if not whole or whole & test_unknown or len(test_unknown) + len(whole) > cross_target:
                continue
            try_add(whole, to_val=False)

    for sku in order:
        if len(test_unknown) >= config.num_test_unknown:
            break
        if sku in val_unknown or sku in test_unknown:
            continue
        family = family_of.get(sku, sku)
        if cross_target:
            unknown_now = val_unknown | test_unknown | {sku}
            if not (members[family] - unknown_now):
                continue  # would remove the family's last catalogue SKU
        try_add({sku}, to_val=False)

    if len(val_unknown) < config.num_val_unknown or len(test_unknown) < config.num_test_unknown:
        raise ValueError(
            f"Could hold out only {len(val_unknown)} validation-unknown and {len(test_unknown)} "
            "test-unknown SKUs while every catalogue SKU keeps two independent photographs"
        )
    return val_unknown, test_unknown


def build_evaluation_split(
    samples: Sequence[Sample],
    config: SplitConfig | None = None,
    content_hash: Callable[[Sample], str] | None = None,
) -> SplitResult:
    """Build the five evaluation roles from labelled crops with source groups.

    Args:
        samples: Every crop, with ``sku_id``, ``family_id`` and ``source_group``.
        config: Split settings.
        content_hash: Optional function returning a hash of a crop's file content; when given,
            queries byte-identical to a gallery image are removed.
    """
    config = config or SplitConfig()
    rng = random.Random(config.seed)

    by_group: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_group[sample.source_group].append(sample)
    groups_of: dict[str, set[str]] = defaultdict(set)
    for group, members in by_group.items():
        for sample in members:
            groups_of[sample.sku_id].add(group)

    family_of = {s.sku_id: s.family_id for s in samples if s.family_id}
    val_unknown, test_unknown = _draw_unknowns(
        sorted(groups_of), groups_of, family_of, config, rng
    )
    unknown = val_unknown | test_unknown

    roles: dict[str, list[Sample]] = {role: [] for role in ROLES}
    removed_known = 0
    known_groups: list[str] = []
    groups = sorted(by_group)
    rng.shuffle(groups)
    for group in groups:
        present = {s.sku_id for s in by_group[group]}
        if present & test_unknown:
            role, keep = "test_unknown", test_unknown
        elif present & val_unknown:
            role, keep = "val_unknown", val_unknown
        else:
            known_groups.append(group)
            continue
        roles[role].extend(s for s in by_group[group] if s.sku_id in keep)
        removed_known += sum(1 for s in by_group[group] if s.sku_id not in unknown)

    # Gallery: per SKU, draw photographs until the SKU's gallery target is met.
    sku_groups: dict[str, list[str]] = defaultdict(list)
    sku_total: dict[str, int] = defaultdict(int)
    for group in known_groups:
        for sample in by_group[group]:
            sku_total[sample.sku_id] += 1
        for sku in sorted({s.sku_id for s in by_group[group]}):
            sku_groups[sku].append(group)
    assigned: dict[str, str] = {}
    gallery_count: dict[str, int] = defaultdict(int)
    # Photographs of each SKU that may still become test queries.
    open_groups: dict[str, int] = {sku: len(groups) for sku, groups in sku_groups.items()}
    skus_of = {group: {s.sku_id for s in by_group[group]} for group in known_groups}

    def take(group: str, role: str) -> bool:
        """Assign a photograph unless it would leave one of its SKUs without a test photograph."""
        if group in assigned or any(open_groups[s] <= 1 for s in skus_of[group]):
            return False
        assigned[group] = role
        for sku in skus_of[group]:
            open_groups[sku] -= 1
        return True

    catalogue = sorted(sku_groups)
    rng.shuffle(catalogue)
    for sku in catalogue:
        target = min(
            config.max_gallery_per_sku,
            max(1, math.floor(sku_total[sku] * config.gallery_fraction)),
        )
        for group in sku_groups[sku]:
            if gallery_count[sku] >= target:
                break
            if take(group, "gallery"):
                for sample in by_group[group]:
                    gallery_count[sample.sku_id] += 1

    for group in known_groups:
        if group not in assigned and rng.random() < config.val_fraction:
            take(group, "val_known")
    for group in known_groups:
        roles[assigned.get(group, "test_known")].extend(by_group[group])

    removed_duplicates = 0
    if content_hash is not None:
        gallery_hashes = {content_hash(s) for s in roles["gallery"]}
        for role in ("val_known", "test_known"):
            kept = [s for s in roles[role] if content_hash(s) not in gallery_hashes]
            removed_duplicates += len(roles[role]) - len(kept)
            roles[role] = kept

    usable = {s.sku_id for s in roles["gallery"]} & {s.sku_id for s in roles["test_known"]}
    unused = sorted(set(sku_groups) - usable)
    for role in ("gallery", "val_known", "test_known"):
        roles[role] = [s for s in roles[role] if s.sku_id in usable]
    for role in roles:
        roles[role].sort(key=lambda s: (s.sku_id, s.path))
    return SplitResult(roles, removed_known, removed_duplicates, unused)
