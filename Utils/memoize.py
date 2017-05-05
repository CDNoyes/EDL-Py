from functools import wraps

def memoize(function):
    """ A python 2.7 solution for memoization of functions """

    memo = {}

    @wraps(function)
    def wrapper(*args):
        args_key = tuple(tuple(arg) if hasattr(arg,'__iter__') else arg for arg in args)
        if args_key in memo:
            return memo[args_key]
        else:
            rv = function(*args)
            memo[args_key] = rv
            return rv
    return wrapper