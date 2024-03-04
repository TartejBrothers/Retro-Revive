import cv2
import os
import numpy as np


def split_video_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        frame_name = f"frame_{frame_count:04d}.jpeg"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    video_capture.release()


def generate_video3(image_folder, output_folder, input_video_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = [
        img
        for img in os.listdir(image_folder)
        if img.endswith((".png", ".jpg", ".jpeg"))
    ]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video_name = os.path.join(output_folder, "outputvideo.mp4")

    # Get the original frame rate of the input video
    video_capture = cv2.VideoCapture(input_video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_capture.release()

    # Create VideoWriter object with the original frame rate
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Specify the codec (H.264)
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Write colorized frames to video
    for i in range(frame_count):
        frame_path = os.path.join(image_folder, f"frame_{i:04d}.jpeg")
        colorized_frame = colorize_image(frame_path, net)
        if colorized_frame is not None:
            video.write(colorized_frame)

    cv2.destroyAllWindows()
    video.release()


def colorize_image(image_path, net):
    bw_image = cv2.imread(image_path)
    if bw_image is None:
        print("Error: Unable to read image from", image_path)
        return None

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

    return colorized


if __name__ == "__main__":
    input_video_path = "video/test.mp4"
    output_frames_folder = "output_frame"
    output_video_path = "outputvideo"

    prototxt_path = "colorization_deploy_v2.prototxt"
    model_path = "colorization_release_v2.caffemodel"
    kernel_path = "pts_in_hull.npy"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)
    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
        np.full([1, 313], 2.606, dtype="float32")
    ]

    split_video_frames(input_video_path, output_frames_folder)
    generate_video3(output_frames_folder, output_video_path, input_video_path)
