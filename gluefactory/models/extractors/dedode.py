import kornia
import torch

from ..base_model import BaseModel
from ..utils.misc import pad_and_stack
from DeDoDe import dedode_detector_L, dedode_descriptor_B, dedode_descriptor_G


class DeDoDe(BaseModel):
    default_conf = {
        "descriptor": "dedode_descriptor_B.pth",
        "detector": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth",
        "dense_outputs": False,
        "max_num_keypoints": None,
        "desc_dim": 256,
        "resize_to": 14*56,
        "detection_threshold": 0.0,
        "force_num_keypoints": False,
        "pad_if_not_divisible": True,
        "chunk": 4,  # for reduced VRAM in training
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        self.detector =  dedode_detector_L(weights =  torch.hub.load_state_dict_from_url(conf["detector"], map_location=torch.device('cpu')))
        if conf.descriptor == 'dedode_descriptor_G.pth':
            self.descriptor =  dedode_descriptor_G(weights = torch.load("dedode_descriptor_G.pth"))
        else:
            self.descriptor =  dedode_descriptor_B(weights = torch.load("dedode_descriptor_B.pth"))
        self.set_initialized()

    def _forward(self, data):
        image = data["image"]
        
        B, ch, H, W = image.shape
        timg_resized = kornia.geometry.resize(image, (self.conf.resize_to, self.conf.resize_to), antialias=True)
        h, w = timg_resized.shape[2:]
        keypoints, scores, descriptors = [], [], []
        chunk = self.conf.chunk
        for i in range(0, image.shape[0], chunk):
            with torch.inference_mode():
                detections_A = self.detector.detect({"image": self.detector.normalizer(timg_resized[i: min(image.shape[0], i + chunk)])},
                    num_keypoints = self.conf.max_num_keypoints)
            keypoints_A, _ = detections_A["keypoints"], detections_A["confidence"]
            keypoints_A_px = self.detector.to_pixel_coords(keypoints_A, H, W)
            with torch.inference_mode():
                descs = self.descriptor.describe_keypoints({"image": self.descriptor.normalizer(timg_resized[i: min(image.shape[0], i + chunk)])},
                                                  keypoints_A)['descriptions']
            DDIM = descs.shape[-1]
            keypoints += [x for x in keypoints_A_px]
            scores += [x  for x in detections_A["confidence"]]
            descriptors += [f for f in descs]
            #del keypoints_A

        pred = {
            "keypoints": torch.stack(keypoints).reshape(B, -1, 2),
            "keypoint_scores": torch.stack(scores).reshape(B, -1),
            "descriptors": torch.stack(descriptors).reshape(B, -1, DDIM),
        }
        #for k, v in pred.items():
        #    print (k, v.shape)
        return pred

    def loss(self, pred, data):
        raise NotImplementedError
