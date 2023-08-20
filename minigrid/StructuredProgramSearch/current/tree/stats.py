import numpy as np

findobj_time = np.array([596.63 + 5.236, 540.33 + 6.2311, 951.17 + 5.51707])

print(np.mean(findobj_time))
print(np.std(findobj_time))

multi_time = np.array([5.049, 5.008, 4.918])
print(np.mean(multi_time))
print(np.std(multi_time))

locked_time = np.array([28*60 + 54.96, 26*60 + 19.91, 25*60 + 4.65])
print(np.mean(locked_time))
print(np.std(locked_time))