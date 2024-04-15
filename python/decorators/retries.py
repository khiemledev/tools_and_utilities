import time
from functools import wraps


def retries(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error: {e}")
                    if i < max_retries - 1:
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        raise

        return wrapper

    return decorator


n_try = 0


def main():

    @retries()
    def my_func():
        global n_try
        print("This is the {} try".format(n_try))

        if n_try == 3 or n_try == 4:
            n_try += 1
            raise Exception("Fake exception to test retry")

        n_try += 1

    for i in range(10):
        my_func()


if __name__ == "__main__":
    main()
