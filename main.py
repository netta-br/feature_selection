"""Feature selection pipeline entry point."""
import sys


def main():
    from feature_selection.src.pipeline.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
