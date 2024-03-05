from django.shortcuts import render
from django.template.response import TemplateResponse
from django.core.files.storage import default_storage
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from django.utils.datastructures import MultiValueDictKeyError
import os
import cv2
import numpy as np
from skimage.io import imsave
from skimage import img_as_ubyte
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from django.http import JsonResponse


class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        return name


def index(request):
    message = ""
    fss = CustomFileSystemStorage()

    try:
        media_root = settings.MEDIA_ROOT
        for filename in os.listdir(media_root):
            file_path = os.path.join(media_root, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                return JsonResponse(
                    {"error": f"Failed to delete {filename}: {str(e)}"}, status=500
                )
        prototxt_path = os.path.join(
            settings.BASE_DIR, "requirements", "colorization_deploy_v2.prototxt"
        )
        model_path = os.path.join(
            settings.BASE_DIR, "requirements", "colorization_release_v2.caffemodel"
        )
        kernel_path = os.path.join(settings.BASE_DIR, "requirements", "pts_in_hull.npy")
        output_folder = os.path.join(settings.BASE_DIR, "output")

        image = request.FILES["image"]
        _image = fss.save(image.name, image)

        path = os.path.join(settings.MEDIA_ROOT, _image)
        bw_image = cv2.imread(path)

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

        # cv2.imwrite("colorized_image.jpg", colorized)
        result_image_name = "img_result.png"

        result_image_path = os.path.join(settings.MEDIA_ROOT, result_image_name)

        filename = _image
        result_image_url = os.path.join(settings.MEDIA_URL, result_image_name)

        cv2.imwrite(result_image_path, colorized)

        return TemplateResponse(
            request,
            "colorised.html",
            {
                "message": message,
                "image_url": fss.url(_image),
                "result_image_url": result_image_url,
            },
        )
    except MultiValueDictKeyError:
        return TemplateResponse(
            request,
            "colorised.html",
            {"message": "No Image Selected"},
        )
    except Exception as e:
        return TemplateResponse(
            request,
            "colorised.html",
            {"message": str(e)},
        )


def video(request):
    message = ""
    fss = FileSystemStorage()

    try:
        media_root = settings.MEDIA_ROOT
        for filename in os.listdir(media_root):
            file_path = os.path.join(media_root, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                return JsonResponse(
                    {"error": f"Failed to delete {filename}: {str(e)}"}, status=500
                )
        media_root = os.path.join(settings.BASE_DIR, "output_frames")
        for filename in os.listdir(media_root):
            file_path = os.path.join(media_root, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                return JsonResponse(
                    {"error": f"Failed to delete {filename}: {str(e)}"}, status=500
                )

        prototxt_path = os.path.join(
            settings.BASE_DIR, "requirements", "colorization_deploy_v2.prototxt"
        )
        model_path = os.path.join(
            settings.BASE_DIR, "requirements", "colorization_release_v2.caffemodel"
        )
        kernel_path = os.path.join(settings.BASE_DIR, "requirements", "pts_in_hull.npy")
        output_folder = os.path.join(settings.BASE_DIR, "media")

        video = request.FILES["video"]
        video_path = fss.save(video.name, video)
        video_full_path = os.path.join(settings.MEDIA_ROOT, video_path)

        output_frames_folder = os.path.join(settings.BASE_DIR, "output_frames")
        output_video_path = os.path.join(settings.MEDIA_ROOT)

        split_video_frames(video_full_path, output_frames_folder)
        generate_video(
            output_frames_folder,
            output_folder,
            video_full_path,
            prototxt_path,
            model_path,
            kernel_path,
        )

        # Construct the video URLs
        video_url = fss.url(video_path)

        result_video_url = os.path.join(settings.MEDIA_URL, "output_video.mp4")

        print("Output Video Path:", result_video_url)
        
        media_root = os.path.join(settings.BASE_DIR, "output_frames")
        for filename in os.listdir(media_root):
            file_path = os.path.join(media_root, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                return JsonResponse(
                    {"error": f"Failed to delete {filename}: {str(e)}"}, status=500
                )
        
        return TemplateResponse(
            request,
            "colorised-video.html",
            {
                "message": message,
                "video_url": video_url,
                "result_video_url": result_video_url,
            },
        )
    except MultiValueDictKeyError:
        return TemplateResponse(
            request, "colorised-video.html", {"message": "No Video Selected"}
        )
    except Exception as e:
        return TemplateResponse(request, "colorised-video.html", {"message": str(e)})


def split_video_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)
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


def generate_video(
    image_folder,
    output_folder,  # Change the output_folder parameter name
    input_video_path,
    prototxt_path,
    model_path,
    kernel_path,
):
    print("Generate is Working")
    if not os.path.exists(image_folder):
        return

    images = [
        img
        for img in os.listdir(image_folder)
        if img.endswith((".png", ".jpg", ".jpeg"))
    ]
    if not images:
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    # Change the output_video_path to use the output_folder
    output_video_path = os.path.join(output_folder, "output_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Specify the codec (H.264)
    video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)
    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
        np.full([1, 313], 2.606, dtype="float32")
    ]

    for img in sorted(images):
        frame_path = os.path.join(image_folder, img)
        colorized_frame = colorize_image(frame_path, net)
        if colorized_frame is not None:
            video.write(colorized_frame)

    cv2.destroyAllWindows()
    video.release()


def colorize_image(image_path, net):
    bw_image = cv2.imread(image_path)
    if bw_image is None:
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
