from functools import wraps
from time import perf_counter
from typing import Callable


def timeit(func: Callable) -> None:
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time

        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} took {end - start:.5f} seconds")
        return result

    return wrapper


@timeit
def fibonacci(n: int) -> int:
    if n == 1 or n == 2:
        return 1

    memory = [1] * n
    for i in range(3, n + 1):
        memory[i - 1] = memory[i - 2] + memory[i - 3]
    return memory[n - 1]


def main():
    fibonacci(100)
    fibonacci(int(10**5 * 2))


if __name__ == "__main__":
    main()
