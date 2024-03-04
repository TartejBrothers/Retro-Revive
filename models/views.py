from .colorised import colorize_image
import cv2
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.shortcuts import render, redirect


def index(request):
    if request.method == "POST" and request.FILES["image"]:
        try:
            image = request.FILES["image"]
            _image = default_storage.save(image.name, image)
            path = os.path.join(settings.MEDIA_ROOT, _image)
            colorized_image = colorize_image(path)

            colorized_filename = f"colorized_{_image}"
            colorized_path = os.path.join(settings.MEDIA_ROOT, colorized_filename)
            cv2.imwrite(colorized_path, colorized_image)

            return JsonResponse(
                {"colorized_image_url": default_storage.url(colorized_filename)}
            )
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return render(request, "colorised.html")
