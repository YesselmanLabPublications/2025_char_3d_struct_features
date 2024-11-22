from typing import List
from dms_quant_framework.logger import get_logger

log = get_logger("util")


def find_stretches(nums: List[int]) -> List[List[int]]:
    """Finds all consecutive number stretches in a list of integers.

    Args:
        nums (List[int]): A list of integers that may contain consecutive numbers.

    Returns:
        List[List[int]]: A list of lists, each containing the start and end of a
        consecutive number stretch.

    Raises:
        ValueError: If `nums` contains non-integer elements.

    Example:
        >>> find_stretches([3, 4, 5, 10, 11, 12])
        [[3, 5], [10, 12]]

        >>> find_stretches([1, 2, 3, 7, 8, 10])
        [[1, 3], [7, 8], [10, 10]]

    Notes:
        The input list is sorted within the function to simplify the logic for
        finding consecutive stretches.
    """

    log.debug("Initial list: %s", nums)

    if len(nums) == 0:
        return []

    nums = sorted(set(nums))
    log.debug("Sorted and de-duplicated list: %s", nums)

    stretches = []
    start = end = nums[0]

    for num in nums[1:]:
        if num == end + 1:
            end = num
        else:
            stretches.append([start, end])
            start = end = num

    stretches.append([start, end])
    log.debug("Identified stretches: %s", stretches)
    return stretches
