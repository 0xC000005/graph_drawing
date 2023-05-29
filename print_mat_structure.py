import scipy.io
import numpy as np

def print_mat_file(filename, level=0):
    # Load .mat file
    mat = scipy.io.loadmat(filename)

    def print_item(key, item, level=0):
        # Print indentation
        print("  " * level + f"{key}: ", end="")
        if isinstance(item, np.ndarray) and item.size > 0:
            if item.dtype == 'object':
                print(f"type({type(item)}), shape({item.shape})")
                for i, subitem in np.ndenumerate(item):
                    print_item(i, subitem, level + 1)
            else:
                print(f"type({type(item)}), value({item})")
        else:
            print(f"{item}")

    # Print structure and data
    for key, value in mat.items():
        print_item(key, value, level)

print_mat_file('input_graphs/SuiteSparse Matrix Collection/dwt_307.mat')
