import os
from pathlib import Path

def split_data(path, target_path):

    train_split = 0.75
    test_split = 0.15
    valid_split = 0.10

    
    if Path(path).is_dir():
        lines = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)

            with open(file_path, "r") as f:
                lines.extend(f.readlines())
    else:
        with open(path) as f:
            lines = f.readlines()

    total_lines = len(lines) * 0.1

    train_slice = total_lines * train_split
    test_slice = train_slice + (test_split * total_lines)
    valid_slice = test_slice + (valid_split * total_lines)

    train_lines = lines[:train_slice]
    test_lines = lines[train_slice:test_slice]
    valid_lines = lines[test_slice:valid_slice]

    with open(target_path + "/train.txt", "w") as f:
        f.write("\n".join(train_lines))

    with open(target_path + "/test.txt", "w") as f:
        f.write("\n".join(test_lines))

    with open(target_path + "/valid.txt", "w") as f:
        f.write("\n".join(valid_lines))


if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 3

    split_data(sys.argv[1], sys.argv[2])
