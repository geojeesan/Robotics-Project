import numpy as np
map= np.load(r"controllers\task_allocator_v2\final_map.npy")
print(f"unique numbers: {np.unique(map)}")
x=np.unique_counts(map)
print(f" uniques: {x}")
print(map[189][153])


 