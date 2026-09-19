"""Load and cache the CaseHOLD dataset."""

from datasets import load_dataset


def load_casehold():
    """Load CaseHOLD dataset from HuggingFace (via lex_glue).

    Returns DatasetDict with train/validation/test splits.
    Each sample has:
        - context: citing context with <HOLDING> placeholder
        - endings: list of 5 candidate holdings
        - label: index (0-4) of the correct holding
    """
    return load_dataset("lex_glue", "case_hold")


if __name__ == "__main__":
    ds = load_casehold()
    print(ds)
    print(f"\nTrain example:\n{ds['train'][0]}")
