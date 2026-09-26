import hashlib


def parse_matches(value):
    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def match_count_bucket(n):
    """
    We don't need to distinguish very large match counts
    for stratification purposes.
    """

    if n == 0:
        return "0"

    if n == 1:
        return "1"

    if n == 2:
        return "2"

    if n == 3:
        return "3"

    if n == 4:
        return "4"

    return "5+"


def deterministic_fraction(value, seed=42):
    """
    Deterministically maps a string to a number in [0, 1).

    This avoids relying on Python's built-in hash(), which can
    change between processes.
    """

    text = f"{seed}:{value}"

    digest = hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()

    integer = int(
        digest[:16],
        16
    )

    return integer / float(16 ** 16)


def assign_split(
    source1_entity_id,
    validation_fraction=0.20,
    seed=42,
):
    value = deterministic_fraction(
        source1_entity_id,
        seed=seed,
    )

    if value < validation_fraction:
        return "val"

    return "train"