import os
import shutil


def rename():
    files = sorted(os.listdir("frames/"))
    count = 1
    for f in files:
        source_f = f"frames/{f}"
        dest_f = f"frames/img{count:06}.png"
        shutil.move(source_f, dest_f)
        # os.remove(source_f)
        count += 1


if __name__ == "__main__":
    rename()
