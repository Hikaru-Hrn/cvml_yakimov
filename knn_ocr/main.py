import cv2
import numpy as np
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread
from scipy.ndimage import binary_closing


test_path = Path("./task")
train_path = test_path / "train"

def extractor(image):
    if image.ndim == 2:
        binary = image
    else:
        gray = np.mean(image, 2).astype("u1")
        binary = gray > 0
    lb = label(binary)
    props = regionprops(lb)[0]

    return np.array([*props.moments_hu, props.eccentricity])    

def make_train(path):
    train, responses = [], []
    for cls in sorted(path.glob("*")):
        name = cls.name
        if len(name) == 1:
            ncls = ord(name)
        else:
            ncls = ord(name[1])

        for i in cls.glob("*.png"):
            train.append(extractor(imread(i)))
            responses.append(ncls)
            
    train = np.array(train, dtype="f4")
    responses = np.array(responses, dtype="f4")
    return train, responses


train, responses = make_train(train_path)
knn = cv2.ml.KNearest.create()
knn.train(train, cv2.ml.ROW_SAMPLE, responses)

struct = np.ones((18, 1))

for img in range(0, 6+1):
    image = imread(test_path / f"{img}.png")
    gray = image.mean(2)
    binary = gray > 0

    binary_closed = binary_closing(binary, structure=struct)
    lb = label(binary_closed)
    props = regionprops(lb)
    sorted_props = sorted(props, key=lambda prop: prop.centroid[1])

    answer = ""
    for i in range(len(sorted_props)):
        if sorted_props[i].area < 300:
            continue

        y1, x1, y2, x2 = sorted_props[i].bbox
        if i != 0:
            _, _, _, prev_x2 = sorted_props[i-1].bbox

            if x1 - prev_x2 > 25:
                answer += " "

#        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
#        cv2.putText(image, str(len(answer)), (x1, y1), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        roi = binary[y1: y2, x1:x2]
        find = np.array([extractor(roi)], dtype="f4")
        ret, results, neighbor, dist = knn.findNearest(find, k=5)
        answer += chr(int(results[0][0]))
#        cv2.imshow(f"Debug Image {img}", image)
#        cv2.waitKey(0)
    print(f"Результат для изображения {img}: {answer}")
#    cv2.destroyAllWindows()











