import scipy.io


def print_mat_file(filename):
    # Load .mat file
    mat = scipy.io.loadmat(filename)

    # Print top-level structure
    for key, value in mat.items():
        if key in ["__header__", "__version__", "__globals__"]:
            print(f"{key}: {value}")
        else:
            print(f"{key}: type({type(value)}), shape({value.shape})")

            # If value is numpy ndarray and contains dtype object, it can have nested structures.
            if isinstance(value, np.ndarray) and value.dtype == 'object':
                for i, item in np.ndenumerate(value):
                    print(f"  {i}:", )
                    if isinstance(item, np.ndarray) and item.size > 0:
                        print(f"    type({type(item[0])}), value({item[0]})")
                    else:
                        print(f"    {item}")


print_mat_file('input_graphs/SuiteSparse Matrix Collection/dwt_307.mat')
