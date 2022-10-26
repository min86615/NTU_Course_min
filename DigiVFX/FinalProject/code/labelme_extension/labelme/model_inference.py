from labelme.mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2
import json
import base64
import torch

class LabelFileError(Exception):
    pass

def mask_area(result, threshold_score):
    list_shape_point = []
    list_label_idx = []
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        for idx_type, type_result in enumerate(bbox_result):
            if len(type_result) > 0:
                for idx_bbox_result, bbox_result in enumerate(type_result):
                    if bbox_result[-1] > threshold_score:
                        list_label_idx.append(idx_type)
                        predict_mask = (segm_result[idx_type][idx_bbox_result]*255).astype("uint8")
                        cnts, hierarchy = cv2.findContours(predict_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
                        list_point = []
                        len_cnts = len(cnts)
                        if len_cnts > 1:
                            idx_max_area = 0
                            tmp_area = 0
                            for idx_cnts, contours in enumerate(cnts):
                                _areasize = cv2.contourArea(contours)
                                if _areasize > tmp_area:
                                    idx_max_area = idx_cnts
                                    tmp_area = _areasize
                            for point in cnts[idx_max_area]:
                                list_point.append(point[0].tolist())
                        else:
                            for point in cnts[0]:
                                list_point.append(point[0].tolist())
                        list_shape_point.append(list_point)
    return list_label_idx, list_shape_point

def inference_image(img_path, model, threshold_score=0.3):
    # config_file = 'labelme\modelconfig\solov2\solov2_r50_fpn_3x_coco.py'
    # checkpoint_file = 'solov2_r50_fpn_3x_coco_20220512_125856-fed092d4.pth'
    # if torch.cuda.is_available():
    #     print("Current Use GPU")
    #     device = "cuda:0"
    # else:
    #     print("Current Use CPU")
    #     device = "cpu"
    # model = init_detector(config_file, checkpoint_file, device=device)
    img = cv2.imread(img_path)
    img_h, img_w, img_c = img.shape
    with open(img_path, "rb") as image_file:
        str_encoded = base64.b64encode(image_file.read())
    img_data = str_encoded.decode("utf-8")
    result = inference_detector(model, img_path)
    list_label_idx, list_shape_point = mask_area(result, threshold_score)
    list_dict_shape = []
    for label_idx, shape_point in zip(list_label_idx, list_shape_point):
        dict_shape = dict(
            label = model.CLASSES[label_idx],
            points = shape_point,
            group_id = None,
            shape_type = "polygon",
            flag = {},
        )
        list_dict_shape.append(dict_shape)
    data = dict(
        version = "5.0.1",
        flag = {},
        shapes = list_dict_shape,
        imagePath = img_path.split("\\")[-1],
        imageData = img_data,
        imageHeight = img_h,
        imageWidth = img_w,
    )
    try:
        with open("%s.json" % (img_path.split(".")[0]), "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise LabelFileError(e)