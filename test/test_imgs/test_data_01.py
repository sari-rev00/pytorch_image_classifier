import os

BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "img")
)

labeling = {
    "car": 0,
    "bike": 1
}

img_format = ['jpg']

def get_test_datainfo():
    fpaths = list()
    labels = list()

    for dir in os.listdir(BASE_PATH):
        if dir in labeling.keys():
            l = labeling[dir]
            for fname in os.listdir(os.path.join(BASE_PATH, dir)):
                if not fname.split('.')[-1] in img_format:
                    continue
                fpaths.append(os.path.join(BASE_PATH, dir, fname))
                labels.append(l)
    return fpaths, labels


if __name__ == '__main__':
    fpaths, labels = get_test_datainfo()
    for f, l in zip(fpaths, labels):
        print(f"{f}: {l}")


