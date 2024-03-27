import numpy as np
import torch


def cam(features, weights, class_id):
    # features is a tensor with [b, c, h, w]
    b, c, h, w = features.size()

    # weights should be a list (in fc, the weights is each class weights)
    output_cam = []
    for idx in class_id:
        cam = weights[idx].dot(features.reshape((c, h * w)))
        cam = cam.reshape((h, w))
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())
        cam_img = np.array(255 * cam_img).astype(np.uint8)
        output_cam.append(cam_img)
    return output_cam


if __name__ == '__main__':
    features = torch.randn(1, 128, 7, 7)
    weights = torch.nn.Linear(in_features=128, out_features=10).weight.detach().numpy()
    class_id = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    cams = cam(features, weights, class_id)
    print(cams)
