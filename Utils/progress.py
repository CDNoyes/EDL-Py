


def progress(current, total, percent=10, iteration=None):
    """
        Used in a loop to indicate progress
    """
    current += 1
    if current:
        previous = current - 1
    else:
        previous = current 

    # print out every percent
    frac = percent/100. 
    value = max(1, frac*total)

    return not (int(current/value) == int(previous/value))

        


if __name__ == "__main__":
    for i in range(17):
        print(i)
        if progress(i, 17):
            print(r"Another 10% completed")