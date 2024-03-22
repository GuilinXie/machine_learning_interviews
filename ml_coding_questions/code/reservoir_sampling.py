import random

def reservoir_sampling(stream, k):
    result = []
    cnt = 0
    for item in stream:
        cnt = cnt + 1
        if len(result) < k:
            result.append(item)
        else:
            j = random.randint(0, cnt - 1)          # a random number in [0, cnt]
            if j < k:
                result[j] = item
    return result


if __name__ == "__main__":
    stream = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    k = 5
    print(reservoir_sampling(stream, k))


            
    