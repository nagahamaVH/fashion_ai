import numpy as np
from PIL import Image
import cv2


def draw_bbox(image, info, classes):
    pass


def draw_segm(image, info, classes):
    pass


def draw_contour(image, segm, color):
    ALPHA = 0.6

    image_proc = image.copy()
    segm_mask = np.zeros_like(image, np.uint8)
    segm = np.uint8(segm)

    # Improving segmentation draw (comment this block to use every pixel)
    _, segm = cv2.threshold(segm, 127, 255, cv2.THRESH_BINARY_INV)
    segm = cv2.bitwise_not(segm)

    contours, _ = cv2.findContours(segm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for _, contour in enumerate(contours):
        # Draw polygon on segmentation mask
        cv2.fillPoly(segm_mask, pts=[contour], color=color)

        # Merging proc image and segmentation
        mask = segm.astype(bool)
        image_proc[mask] = cv2.addWeighted(image_proc, ALPHA, segm_mask, 1 - ALPHA, 0)[mask]

        # Add contours to proc image
        cv2.drawContours(image_proc, [contour], -1, color, 1)
    return image_proc


if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    from predict import predict

    image_path = "../data/train/0a0f94b4e785bb2326dd1832303ce8de.jpg"
    # image_path = "lixo.jpeg"
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256), resample=Image.BILINEAR)
    image = np.asarray(image)

    info = predict(image, "test_20epoch")

    segm = info["masks"][1, 0]
    proc_image = draw_contour(image, segm, (255, 0, 0))

    plt.imshow(proc_image)
    plt.show()

    plt.imshow(segm)
    plt.show()
