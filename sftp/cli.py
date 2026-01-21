"""Command-line interface for SpanFinder."""

import argparse
import json
import sys

from . import __version__, load_model


def main():
    """Main entry point for the span-finder CLI."""
    parser = argparse.ArgumentParser(
        prog="span-finder",
        description="SpanFinder: Parse sentences by finding & labeling spans",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"span-finder {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict spans on sentences")
    predict_parser.add_argument(
        "sentence",
        nargs="?",
        help="Sentence to parse (if not provided, reads from stdin)",
    )
    predict_parser.add_argument(
        "-m", "--model",
        default="https://public.gqin.me/framenet/20210127.fn.tar.gz",
        help="Path or URL to model checkpoint",
    )
    predict_parser.add_argument(
        "-d", "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu)",
    )
    predict_parser.add_argument(
        "-f", "--format",
        default="tree",
        choices=["tree", "json"],
        help="Output format (default: tree)",
    )
    predict_parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index to use (e.g., --gpu 0)",
    )

    args = parser.parse_args()

    if args.command == "predict":
        run_predict(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_predict(args):
    """Run prediction on a sentence."""
    # Determine device
    device = args.device
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"

    # Get sentence
    if args.sentence:
        sentence = args.sentence
    else:
        # Read from stdin
        sentence = sys.stdin.read().strip()
        if not sentence:
            print("Error: No sentence provided", file=sys.stderr)
            sys.exit(1)

    # Load model and predict
    print(f"Loading model from {args.model}...", file=sys.stderr)
    predictor = load_model(model_path=args.model, device=device)

    print(f"Predicting...", file=sys.stderr)
    result = predictor.predict_sentence(sentence)

    # Output result
    if args.format == "tree":
        result.span.tree(result.sentence)
    elif args.format == "json":
        output = {
            "sentence": result.sentence,
            "spans": result.span.to_json(),
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
