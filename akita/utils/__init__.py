import glob
import matplotlib.patches as patches
import numpy as np
import os
import pydicom


from tqdm import tqdm


def get_image(patient_id, image_dir):
    filename = os.path.join(image_dir, f'{patient_id}.dcm')
    dcm_data = pydicom.read_file(filename)
    return dcm_data.pixel_array


def draw_bboxes(bboxes, ax):
    for bbox in bboxes.itertuples():
        rect = patches.Rectangle(
            (bbox.x, bbox.y), bbox.width, bbox.height,
            linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)


def draw_image(df, patient_id, ax, image_dir):
    img = get_image(patient_id, image_dir)
    bboxes = df.loc[df.patientId == patient_id, ['x', 'y', 'width', 'height']]
    draw_bboxes(bboxes, ax)

    ax.imshow(img, cmap='gray')
    ax.grid(False)
    ax.axis('off')


def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))


def parse_dataset(dicom_dir, anns):
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations


# https://goo.gl/cw17ce
# helper function to calculate IoU
def iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union


def map_iou(boxes_true, boxes_pred, scores, thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], min_conf=.98):
    """
    Mean average precision at differnet intersection over union (IoU) threshold

    Args:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    Retruns:
        map: mean average precision of the image
    """

    # According to the introduction, images with no ground truth bboxes will not be
    # included in the map score unless there is a false positive detection (?)
    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None

    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t \
                    and not matched \
                    and j not in matched_bt \
                    and scores[j] > min_conf:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN

        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m

    return map_total / len(thresholds)


def predict_submit(model, image_filepaths, filepath, min_conf=.98):
    data = []
    for image_filepath in tqdm(image_filepaths):
        ds = pydicom.read_file(image_filepath)
        image = ds.pixel_array

        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)

        patient_id = os.path.splitext(os.path.basename(image_filepath))[0]

        results = model.detect([image])
        r = results[0]

        assert(len(r['rois']) == len(r['class_ids']) == len(r['scores']))

        pred_str = ''
        for i in range(len('rois')):
            score = r['scores'][i]
            roi = r['rois'][i]
            pred_str += ' '.join([str(v) for v in [
                score, roi[1], roi[0], roi[3]-roi[1], roi[2]-roi[0]
            ]])
        data.append([patient_id, pred_str])

    df_submit = pd.DataFrame(data=data, columns=['patient_id', 'pred_string'])
    df_submit.to_csv(filepath, index=False)

    return df_submit
