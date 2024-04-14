from functools import cache
from timeit import timeit


@cache
def fibonacci(n: int) -> int:
    if n == 1 or n == 2:
        return 1

    memory = [1] * n
    for i in range(3, n + 1):
        memory[i - 1] = memory[i - 2] + memory[i - 3]
    return memory[n - 1]


def main():
    print(fibonacci.cache_info())

    print("f(100)", fibonacci(100))

    print(fibonacci.cache_info())

    print("f(100)", fibonacci(100))

    print(fibonacci.cache_info())
    fibonacci.cache_clear()

    print(fibonacci.cache_info())


if __name__ == "__main__":
    main()
