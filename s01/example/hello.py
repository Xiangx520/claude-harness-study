"""A small hello-world example."""


def greet(name: str = "World") -> str:
    """Return a friendly greeting for the given name.

    Args:
        name: The person or thing to greet.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"


def main() -> None:
    """Run the command-line greeting."""
    print(greet())


if __name__ == "__main__":
    main()
