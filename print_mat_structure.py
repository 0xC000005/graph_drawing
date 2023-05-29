import scipy.io

# Replace 'file.mat' with your .mat file path
mat_file = scipy.io.loadmat('input_graphs/SuiteSparse Matrix Collection/dwt_307.mat')

# print out all keys/values to view the structure of the file
for key in mat_file:
    print(f"{key}: {mat_file[key]}")
