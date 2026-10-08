import h5py

files = [
    "data/mendeley/test/chestxray14_test.h5",
    "data/mendeley/test/lidcidri2d_test.h5",
    "data/mendeley/test/openbhb2d_test.h5"
]

for file_path in files:

    print("\n" + "=" * 70)
    print(file_path)
    print("=" * 70)

    with h5py.File(file_path, "r") as f:

        def show_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(
                    f"{name} | shape={obj.shape} | dtype={obj.dtype}"
                )
            else:
                print(f"{name}/")

        f.visititems(show_structure)