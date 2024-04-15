from deprecation import deprecated


@deprecated(
    details="Adding ain't cool no more",
    current_version="1.0.2",
    deprecated_in="1.0.0",
    removed_in="1.0.0",
)
def add(x: int, y: int) -> int:
    return x + y


if __name__ == "__main__":
    print(add(5, 7))
