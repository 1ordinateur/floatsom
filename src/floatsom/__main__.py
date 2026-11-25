"""``python -m floatsom``: print the environment check. Exit code 1 means a problem."""

import sys

from .environment import check_environment


def main() -> int:
    report = check_environment(print_report=True)
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
