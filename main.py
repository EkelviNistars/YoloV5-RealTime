from cv2 import rectangle, LINE_AA, getTextSize, putText, imshow, waitKey, destroyAllWindows
from numpy import asarray
from torch import hub
from mss import mss


def plot_one_box(x, img, label=None, line_thickness=1):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = (17, 99, 209)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    rectangle(img, c1, c2, color, thickness=tl, lineType=LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        rectangle(img, c1, c2, color, -1, LINE_AA)
        putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=LINE_AA)


model = hub.load('yolov5', 'yolov5s', source='local')
model.conf = 0.25
model.iou = 0.45
model.agnostic = False
model.max_det = 1000
model.amp = False

with mss() as sct:
    tt = []
    dimensions = {"top": 0, "left": 0, "width": 640, "height": 640}
    for i in range(1000):
        im = asarray(sct.grab(dimensions))
        result = model(im, size=640)
        repos = result.xyxy[0]
        if len(repos) > 0:
            for i in range(len(repos)):
                pos = repos[i].detach()
                plot_one_box(pos, im, label='target')
        imshow('Detect', im)
        if waitKey(1) == 27:
            destroyAllWindows()
            break
