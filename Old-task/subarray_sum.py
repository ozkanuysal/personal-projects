

def get_subarrays(arr):
    subarrays = []
    for i in range(len(arr)):
        for j in range(i, len(arr)):
            subarrays.append(arr[i:j + 1])
    return subarrays


def subarray_sum_slow(arr):
    subarrays = get_subarrays(arr)
    return sum([sum(arr) for arr in subarrays])


def subarray_sum(arr):
    total = 0
    for i, num in enumerate(arr):
        total += num * (i + 1) * (len(arr) - i)
    return total


assert subarray_sum_slow([1]) == subarray_sum([1])
assert subarray_sum_slow([1, 2]) == subarray_sum([1, 2])
assert subarray_sum_slow([1, 2, 3]) == subarray_sum([1, 2, 3])
assert subarray_sum_slow([1, 3, 7]) == subarray_sum([1, 3, 7])
assert subarray_sum_slow([1, 3, 7, 9]) == subarray_sum([1, 3, 7, 9])
assert subarray_sum_slow([1, 3, 7, 9, 11]) == subarray_sum([1, 3, 7, 9, 11])
assert subarray_sum_slow(
    [1, 3, 7, 9, 11, 13]) == subarray_sum([1, 3, 7, 9, 11, 13])