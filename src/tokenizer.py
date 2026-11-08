AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

SEQUENCE_LENGTH = 56

TOKEN_TO_ID = {
    amino_acid: index
    for index, amino_acid in enumerate(AMINO_ACIDS)
}

def encode_sequence(sequence: str) -> list[int]:
    # 将 56 位 GB1 序列编码为氨基酸 ID

    if len(sequence) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected {SEQUENCE_LENGTH} residues, got {len(sequence)}"
        )

    invalid = set(sequence) - set(TOKEN_TO_ID)
    if invalid:
        raise ValueError(f"Unsupported amino acids: {sorted(invalid)}")

    return [TOKEN_TO_ID[amino_acid] for amino_acid in sequence]