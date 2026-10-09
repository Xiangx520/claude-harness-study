def binary_search(sorted_list, target):
	low, high = 0, len(sorted_list) - 1

	while low <= high:
		mid = (low + high) // 2
		mid_value = sorted_list[mid]

		if mid_value == target:
			return mid
		elif mid_value < target:
			low = mid + 1
		else:
			high = mid - 1

	return -1


# Example dictionary: student ID -> name (keys are sorted)
example_dict = {
	101: "Alice",
	102: "Bob",
	103: "Charlie",
	104: "Diana",
	105: "Eve",
}

# Dictionary keys are sorted, so we can binary search on them
keys = list(example_dict.keys())
target_id = 103

index = binary_search(keys, target_id)

if index != -1:
	print(f"Student ID {target_id} found: {example_dict[target_id]}")
else:
	print(f"Student ID {target_id} not found")
