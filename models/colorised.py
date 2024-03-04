import os
import numpy as np
import cv2

input_image = "2.png"


def colorize_image(image_path):
    prototxt_path = "requirements/colorization_deploy_v2.prototxt"
    model_path = "requirements/colorization_release_v2.caffemodel"
    kernel_path = "requirements/pts_in_hull.npy"
    output_folder = "output"  # Specify the output folder

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    bw_image = cv2.imread(image_path)
    if bw_image is None:
        raise ValueError(f"Unable to read image file '{image_path}'.")

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)

    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
        np.full([1, 313], 2.606, dtype="float32")
    ]

    normalized = bw_image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
    L = cv2.split(lab)[0]

    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")

    # Save the colorized image in the output folder
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, colorized)

    return output_path

try:
    colorized_image_path = colorize_image("2.png")
    print("Colorized image saved at:", colorized_image_path)
except Exception as e:
    print("Error:", e)
