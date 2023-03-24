import cv2
from gcloud import bucket
from models import MoistureModel, TKWModel


def get_image():
    img = download_blob()
    return img


def push_results(tkw, moisture):
    raise NotImplementedError


def main():
    image = get_image()
    tkw = TKWModel.predict(image)
    moisture = MoistureModel.predict(image)
    push_results(tkw, moisture)


if __name__ == "__main__":
    main()
