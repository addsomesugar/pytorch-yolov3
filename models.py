import numpy as np

from torch import nn
import torch.nn.functional as F
import torch

from utils.utils import build_targets
from utils.parse_cfg import parse_model_cfg


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    filters = int(hyperparams["channels"])
    output_filters = list()
    output_filters.append(filters)
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2  # pad = int(module_def["pad"])
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = UpSample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class UpSample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()


# def to_cpu(tensor):
#     return tensor.detach().cpu()
#
#
# class YOLOLayer(nn.Module):
#     """Detection layer"""
#
#     def __init__(self, anchors, num_classes, img_dim=416):
#         super(YOLOLayer, self).__init__()
#         self.anchors = anchors
#         self.num_anchors = len(anchors)
#         self.num_classes = num_classes
#         self.ignore_thres = 0.5
#         self.mse_loss = nn.MSELoss()
#         self.bce_loss = nn.BCELoss()
#         self.obj_scale = 1
#         self.noobj_scale = 100
#         self.metrics = {}
#         self.img_dim = img_dim
#         self.grid_size = 0  # grid size
#
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     def compute_grid_offsets(self, grid_size):
#         self.grid_size = grid_size
#         g = self.grid_size
#
#         self.img_stride = self.img_dim / self.grid_size
#         # Calculate offsets for each grid
#         self.grid_x = torch.arange(g, dtype=torch.float).repeat(g, 1).view([1, 1, g, g]).to(device=self.device)
#         self.grid_y = torch.arange(g, dtype=torch.float).repeat(g, 1).t().view([1, 1, g, g]).to(device=self.device)
#         self.scaled_anchors = torch.tensor([(a_w / self.img_stride, a_h / self.img_stride) for a_w, a_h in self.anchors],
#                                       dtype=torch.float).to(device=self.device)
#         self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
#         self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
#
#
#     def forward(self, x, targets=None, img_dim=None):
#
#         self.img_dim = img_dim
#         num_samples = x.size(0)
#         grid_size = x.size(2)
#
#         prediction = (
#             x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
#             .permute(0, 1, 3, 4, 2)
#             .contiguous()
#         )
#
#         # Get outputs
#         x = torch.sigmoid(prediction[..., 0])  # Center x
#         y = torch.sigmoid(prediction[..., 1])  # Center y
#         w = prediction[..., 2]  # Width
#         h = prediction[..., 3]  # Height
#         pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
#         pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
#
#         # If grid size does not match current we compute new offsets
#         if grid_size != self.grid_size:
#             self.compute_grid_offsets(grid_size)
#
#         # Add offset and scale with anchors
#         pred_boxes = torch.zeros_like(prediction[..., :4], dtype=torch.float)
#         pred_boxes[..., 0] = x.data + self.grid_x
#         pred_boxes[..., 1] = y.data + self.grid_y
#         pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
#         pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
#
#         output = torch.cat(
#             (
#                 pred_boxes.view(num_samples, -1, 4) * self.img_stride,
#                 pred_conf.view(num_samples, -1, 1),
#                 pred_cls.view(num_samples, -1, self.num_classes),
#             ),
#             -1,
#         )
#
#         if targets is None:
#             return output, 0
#         else:
#             iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
#                 pred_boxes=pred_boxes,
#                 pred_cls=pred_cls,
#                 target=targets,
#                 anchors=self.scaled_anchors,
#                 ignore_thr=self.ignore_thres
#             )
#
#             obj_mask = obj_mask.long()
#             no_obj_mask = noobj_mask.long()
#             # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
#             loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
#             loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
#             loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
#             loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
#             loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
#             loss_conf_no_obj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
#             loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_no_obj
#             loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
#             total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
#
#             # Metrics
#             cls_acc = 100 * class_mask[obj_mask].mean()
#             conf_obj = pred_conf[obj_mask].mean()
#             conf_no_obj = pred_conf[noobj_mask].mean()
#             conf50 = (pred_conf > 0.5).float()
#             iou50 = (iou_scores > 0.5).float()
#             iou75 = (iou_scores > 0.75).float()
#             detected_mask = conf50 * class_mask * tconf
#             precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
#             recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
#             recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)
#
#             self.metrics = {
#                 "loss": total_loss.detach().cpu().item(),
#                 "x": loss_x.detach().cpu().item(),
#                 "y": loss_y.detach().item(),
#                 "w": loss_w.detach().cpu().item(),
#                 "h": loss_h.detach().cpu().item(),
#                 "conf": loss_conf.detach().cpu().item(),
#                 "cls": loss_cls.detach().cpu().item(),
#                 "cls_acc": cls_acc.detach().cpu().item(),
#                 "recall50": recall50.detach().cpu().item(),
#                 "recall75": recall75.detach().cpu().item(),
#                 "precision": precision.detach().cpu().item(),
#                 "conf_obj": conf_obj.detach().cpu().item(),
#                 "conf_no_obj": conf_no_obj.detach().cpu().item(),
#                 "grid_size": grid_size,
#             }
#
#             return output, total_loss

class YOLOLayer(nn.Module):
    """Detection layer"""
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thr = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.no_obj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 1

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.img_stride, self.grid_x, self.grid_y, self.scaled_anchors, self.anchor_w, self.anchor_h = self.compute_grid_offsets(
            self.grid_size)

    def compute_grid_offsets(self, grid_size):
        self.grid_size = grid_size
        g = self.grid_size

        self.img_stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g, dtype=torch.float, device=self.device).repeat(g, 1).view([1, 1, g, g])
        self.grid_y = torch.arange(g, dtype=torch.float, device=self.device).repeat(g, 1).t().view([1, 1, g, g])
        self.scaled_anchors = torch.tensor([(a_w / self.img_stride, a_h / self.img_stride) for a_w, a_h in self.anchors],
                                      dtype=torch.float).to(device=self.device)
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        return self.img_stride, self.grid_x, self.grid_y, self.scaled_anchors, self.anchor_w, self.anchor_h

    def forward(self, x, targets=None, img_dim=None):

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs [x, y, width, height, confidence, cls_p * 20]
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size)

        # Add offset and scale with anchors
        pred_boxes = pred_boxes = torch.zeros_like(prediction[..., :4], dtype=torch.float, device=self.device)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.img_stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            (iou_scores, class_mask, obj_mask, no_obj_mask,
                true_x, true_y, true_w, true_h, true_cls, true_conf) = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thr=self.ignore_thr
            )
            print(x.size())
            print(pred_cls.size())
            print(true_cls.size())
            print(pred_conf.size())
            print(true_conf.size())
            obj_mask = obj_mask.long()
            no_obj_mask = no_obj_mask.long()
            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], true_x[obj_mask])
            print("x", loss_x.detach().cpu().item())
            loss_y = self.mse_loss(y[obj_mask], true_y[obj_mask])
            print("y", loss_y.detach().cpu().item())
            loss_w = self.mse_loss(w[obj_mask], true_w[obj_mask])
            print("w", loss_w.detach().cpu().item())
            loss_h = self.mse_loss(h[obj_mask], true_h[obj_mask])
            print("h", loss_h.detach().cpu().item())
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], true_conf[obj_mask])
            print("obj", loss_conf_obj.detach().cpu().item())
            loss_conf_no_obj = self.bce_loss(pred_conf[no_obj_mask], true_conf[no_obj_mask])
            print("no_obj", loss_conf_no_obj.detach().cpu().item())
            loss_conf = self.obj_scale * loss_conf_obj + self.no_obj_scale * loss_conf_no_obj
            loss_cls = self.bce_loss(pred_cls[obj_mask], true_cls[obj_mask])
            print("loss_cls", loss_cls.detach().cpu().item())
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            print(total_loss.detach().cpu().item())
            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_no_obj = pred_conf[no_obj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * true_conf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": total_loss.detach().cpu().item(),
                "x": loss_x.detach().cpu().item(),
                "y": loss_y.detach().cpu().item(),
                "w": loss_w.detach().cpu().item(),
                "h": loss_h.detach().cpu().item(),
                "conf": loss_conf.detach().cpu().item(),
                "cls": loss_cls.detach().cpu().item(),
                "cls_acc": cls_acc.detach().cpu().item(),
                "recall50": recall50.detach().cpu().item(),
                "recall75": recall75.detach().cpu().item(),
                "precision": precision.detach().cpu().item(),
                "conf_obj": conf_obj.detach().cpu().item(),
                "conf_no_obj": conf_no_obj.detach().cpu().item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_cfg(cfg_path)
        self.hyper_params, self.module_list = create_modules(self.module_defs)
        self.YOLO_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = torch.cat(yolo_outputs, 1).detach().cpu()
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
