import atexit


@atexit.register
def on_exit_handler() -> None:
    # TODO: do some cleanup here

    print("Have a great time!")


def main():
    print("This is the inner function")

    # try to raise an issue
    1 / 0


if __name__ == "__main__":
    main()
