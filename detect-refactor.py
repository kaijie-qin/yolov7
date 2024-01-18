from numpy import random
from pydantic import BaseModel

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import *
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class Config(BaseModel):
    weights: str = "best.pt"
    img_size: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    device: str = 'cpu'
    view_img: bool = False
    save_txt:bool = False
    save_conf: bool = False
    nosave: bool = False
    classes: int = None
    agnostic_nms: bool = False
    augment: bool = False
    update: bool = False
    project: str = 'runs/detect'
    name: str = "exp"
    exist_ok: bool = False
    no_trace: bool = False
class LogoDetector():
    def init_model(self):
        set_logging()
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.opt.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.opt.img_size, s=self.stride)  # check img_size

        if not self.opt.no_trace:
            self.model = TracedModel(self.model, self.device, self.opt.img_size)

        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()
        return self

    def detect(self, source):

        img0 = cv2.imread(source)  # BGR
        assert img0 is not None, 'Image Not Found ' + source
        img = letterbox(img0, self.opt.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)


        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        self.model(torch.zeros(1, 3, self.opt.img_size, self.opt.img_size).to(self.device).type_as(next(self.model.parameters())))

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.opt.augment)[0]
        t2 = time_synchronized()

            # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        t3 = time_synchronized()

        for path, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)

            cv2.imwrite("/tmp/a.jpg", img0)


    def __init__(self):
        self.opt = Config()
        print(self.opt)


if __name__ == '__main__':
    detector = LogoDetector().init_model()
    detector.detect("cocacola.jpg")
