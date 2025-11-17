import cv2
import glob 
from rembg import remove, new_session
import numpy as np
import glob
from tqdm import tqdm

def background(obj_path, session):
    image = cv2.imread(obj_path)
    result = remove(image, session=session)

    alpha = result[:, :, 3]
    coords = cv2.findNonZero(alpha)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = result[y:y+h, x:x+w]
    return cropped

def resample_prompt(image_obj, obj_size = 40, target = 640):
    h, w = image_obj.shape[:2]
    r = max(h / obj_size, w / obj_size)
    img_small = cv2.resize(image_obj, (int(w / r), int(h // r)), interpolation=cv2.INTER_AREA)
    h, w = img_small.shape[:2]
    
    pad_vert = target - h
    pad_horiz = target - w

    top = pad_vert // 2
    bottom = pad_vert - top
    left = pad_horiz // 2
    right = pad_horiz - left

    if img_small.shape[2] == 4:
        color = (0, 0, 0, 0)
    else:
        color = (0, 0, 0)

    padded = cv2.copyMakeBorder(
        img_small, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )
    return padded

def main():
    session = new_session("birefnet-general")
    folder_path = "/home/quang/DATA/public_test"
    for img_path in tqdm(glob.glob(f"{folder_path}/samples/*/object_images/*.jpg")):
        bg_img = background(img_path, session)
        cv2.imwrite(img_path.replace(".jpg", f"_bg.png"), bg_img)
        for i in [40, 80, 160, 320]:
            target_img = resample_prompt(bg_img, i)
            cv2.imwrite(img_path.replace(".jpg", f"_{i}_resample.png"), target_img)


if __name__ == "__main__":
    main()