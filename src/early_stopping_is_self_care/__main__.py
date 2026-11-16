"""``python -m early_stopping_is_self_care [seed]`` prints a sample roast."""

import sys

from .core import PREFIX, roast


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(PREFIX + roast(seed=seed))


if __name__ == "__main__":
    main()
