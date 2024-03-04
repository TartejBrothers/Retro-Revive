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


class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        return name


def index(request):
    message = ""
    fss = CustomFileSystemStorage()

    try:
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
