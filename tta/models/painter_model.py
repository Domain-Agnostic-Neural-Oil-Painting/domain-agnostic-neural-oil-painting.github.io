import torch
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel
from . import networks
from util import morphology
from PIL import Image
import os


class PainterModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='null')
        parser.add_argument('--used_strokes', type=int, default=8,
                            help='actually generated strokes number')
        parser.add_argument('--num_blocks', type=int, default=3,
                            help='number of transformer blocks for stroke generator')
        parser.add_argument('--lambda_pixel', type=float, default=10.0, help='weight for pixel-level L1 loss')
        parser.add_argument('--lambda_decision', type=float, default=10.0, help='weight for stroke decision loss')
        parser.add_argument('--lambda_recall', type=float, default=10.0, help='weight of recall for stroke decision loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['pixel', 'gt']
        self.visual_names = ['old', 'render', 'rec']
        self.model_names = ['g']
        self.d = 12  # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
        self.d_shape = 5

        def read_img(img_path, img_type='RGB'):
            img = Image.open(img_path).convert(img_type)
            img = np.array(img)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).float() / 255.
            return img
        
        def freeze_non_norm_layers(model):
            for _, module in model.named_modules():
                if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.InstanceNorm2d, torch.nn.GroupNorm)):
                # if isinstance(module, (torch.nn.BatchNorm2d)):
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    for param in module.parameters():
                        param.requires_grad = False

        brush_large_vertical = read_img('brush/brush_large_vertical.png', 'L').to(self.device)
        brush_large_horizontal = read_img('brush/brush_large_horizontal.png', 'L').to(self.device)
        self.meta_brushes = torch.cat(
            [brush_large_vertical, brush_large_horizontal], dim=0)
        net_g = networks.Painter(self.d_shape, opt.used_strokes, opt.ngf,
                                 n_enc_layers=opt.num_blocks, n_dec_layers=opt.num_blocks)
        self.net_g = networks.init_net(net_g, opt.init_type, opt.init_gain, self.gpu_ids)
        # 加载预训练模型
        pretrained_path = 'model.pth'
        if self.isTrain and pretrained_path:
            self.load_pretrained_networks(pretrained_path)
            print(f'Loaded pretrained model from {pretrained_path}')
            
        self.net_g.eval()
        freeze_non_norm_layers(self.net_g)
        self.old = None
        self.render = None
        self.old1 = None
        self.render1 = None
        self.old2 = None
        self.render2 = None
        self.rec = None
        self.gt_param = None
        self.gt_decision = None
        self.pred_param = None
        self.pred_decision = None
        self.patch_size = 32
        self.loss_pixel1 = torch.tensor(0., device=self.device)
        self.loss_pixel2 = torch.tensor(0., device=self.device)
        self.criterion_pixel = torch.nn.MSELoss().to(self.device)
        self.criterion_decision = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(opt.lambda_recall)).to(self.device)
        
        if self.isTrain:
            # self.optimizer = torch.optim.Adam(self.net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.05)
            self.optimizer = torch.optim.AdamW(self.net_g.parameters(), lr=opt.warmup_start_lr, betas=(opt.beta1, 0.999), weight_decay=0.05)
            self.optimizers.append(self.optimizer)
        # # 检查是否正确加载预训练模型    
        # for name, param in self.net_g.named_parameters():
        #     if param.is_floating_point():
        #         print(f"After loading {name}: {param.mean()}")
        
        # self.optimizers=[]
        # if self.isTrain:
        #     params = list(filter(lambda p: p.requires_grad, self.net_g.parameters()))
        #     # print(f"Params to be optimized: {len(params)}")
        #     if len(params) == 0:
        #         raise ValueError("No parameters to optimize")
        #     self.optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
        #     self.optimizers.append(self.optimizer)

    
    def param2stroke(self, param, H, W):
        # param: b, 12
        b = param.shape[0]
        param_list = torch.split(param, 1, dim=1)
        x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
        R0, G0, B0, R2, G2, B2, _ = param_list[5:]
        sin_theta = torch.sin(torch.acos(torch.tensor(-1., device=param.device)) * theta)
        cos_theta = torch.cos(torch.acos(torch.tensor(-1., device=param.device)) * theta)
        index = torch.full((b,), -1, device=param.device)
        index[h > w] = 0
        index[h <= w] = 1
        brush = self.meta_brushes[index.long()]
        alphas = torch.cat([brush, brush, brush], dim=1)
        alphas = (alphas > 0).float()
        t = torch.arange(0, brush.shape[2], device=param.device).unsqueeze(0) / brush.shape[2]
        color_map = torch.stack([R0 * (1 - t) + R2 * t, G0 * (1 - t) + G2 * t, B0 * (1 - t) + B2 * t], dim=1)
        color_map = color_map.unsqueeze(-1).repeat(1, 1, 1, brush.shape[3])
        brush = brush * color_map

        warp_00 = cos_theta / w
        warp_01 = sin_theta * H / (W * w)
        warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
        warp_10 = -sin_theta * W / (H * h)
        warp_11 = cos_theta / h
        warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
        warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
        warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
        warp = torch.stack([warp_0, warp_1], dim=1)
        grid = torch.nn.functional.affine_grid(warp, torch.Size((b, 3, H, W)), align_corners=False)
        brush = torch.nn.functional.grid_sample(brush, grid, align_corners=False)
        alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)

        return brush, alphas

    def set_input(self, input):
        pass

    def process_batch(self, patches, batch_size):
        num_patches = len(patches)
        all_batches = []  # 用于存储所有处理后的批次
        for i in range(0, num_patches, batch_size):
            batch = patches[i:i+batch_size]
            target_canvas = torch.stack([item[0] for item in batch])  # 现在 item[0] 对应 target_canvas
            current_canvas = torch.stack([item[1] for item in batch])  # 现在 item[1] 对应 current_canvas
            all_batches.append((target_canvas, current_canvas))
        return all_batches

    def forward(self):
        
        param1, decisions1 = self.net_g(self.render1, self.old1)
        # stroke_param: b, stroke_per_patch, param_per_stroke
        # decision: b, stroke_per_patch, 1
        self.pred_decision1 = decisions1.view(-1, self.opt.used_strokes).contiguous()
        self.pred_param1 = param1[:, :, :self.d_shape]
        param1 = param1.view(-1, self.d).contiguous()
        foregrounds1, alphas1 = self.param2stroke(param1, self.patch_size, self.patch_size)
        foregrounds1 = morphology.Dilation2d(m=1)(foregrounds1)
        alphas1 = morphology.Erosion2d(m=1)(alphas1)
        # foreground, alpha: b * stroke_per_patch, 3, output_size, output_size
        foregrounds1 = foregrounds1.view(-1, self.opt.used_strokes, 3, self.patch_size, self.patch_size)
        alphas1 = alphas1.view(-1, self.opt.used_strokes, 3, self.patch_size, self.patch_size)
        # foreground, alpha: b, stroke_per_patch, 3, output_size, output_size
        decisions1 = networks.SignWithSigmoidGrad.apply(decisions1.view(-1, self.opt.used_strokes, 1, 1, 1).contiguous())
        self.rec1 = self.old1.clone()
        for j in range(foregrounds1.shape[1]):
            foreground1 = foregrounds1[:, j, :, :, :]
            alpha1 = alphas1[:, j, :, :, :]
            decision1 = decisions1[:, j, :, :, :]
            self.rec1 = foreground1 * alpha1 * decision1 + self.rec1 * (1 - alpha1 * decision1)

        param2, decisions2 = self.net_g(self.render2, self.old2)
        # stroke_param: b, stroke_per_patch, param_per_stroke
        # decision: b, stroke_per_patch, 1
        self.pred_decision2 = decisions2.view(-1, self.opt.used_strokes).contiguous()
        self.pred_param2 = param2[:, :, :self.d_shape]
        param2 = param2.view(-1, self.d).contiguous()
        foregrounds2, alphas2 = self.param2stroke(param2, self.patch_size, self.patch_size)
        foregrounds2 = morphology.Dilation2d(m=1)(foregrounds2)
        alphas2 = morphology.Erosion2d(m=1)(alphas2)
        # foreground, alpha: b * stroke_per_patch, 3, output_size, output_size
        foregrounds2 = foregrounds2.view(-1, self.opt.used_strokes, 3, self.patch_size, self.patch_size)
        alphas2 = alphas2.view(-1, self.opt.used_strokes, 3, self.patch_size, self.patch_size)
        # foreground, alpha: b, stroke_per_patch, 3, output_size, output_size
        decisions2 = networks.SignWithSigmoidGrad.apply(decisions2.view(-1, self.opt.used_strokes, 1, 1, 1).contiguous())
        self.rec2 = self.old2.clone()
        for j in range(foregrounds2.shape[1]):
            foreground2 = foregrounds2[:, j, :, :, :]
            alpha2 = alphas2[:, j, :, :, :]
            decision2 = decisions2[:, j, :, :, :]
            self.rec2 = foreground2 * alpha2 * decision2 + self.rec2 * (1 - alpha2 * decision2)


    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()

        # self.loss_pixel1 = self.criterion_pixel(self.render1, self.rec1) * self.opt.lambda_pixel

        # self.gt_decision = torch.ones(self.opt.batch_size, self.opt.used_strokes, device=self.device)
        # all_pred_decision = self.pred_decision.view(-1).contiguous()
        # self.loss_decision = self.criterion_decision(all_pred_decision, self.gt_decision.view(-1).contiguous()) * self.opt.lambda_decision

        # 将 rec2 和 render2 翻转回去
        rec2_flipped = torch.flip(self.rec2, [3])  # 水平翻转
        render2_flipped = torch.flip(self.render2, [3])
        
        self.loss_pixel1 = self.criterion_pixel(self.render1, self.rec1) * self.opt.lambda_pixel
        self.loss_pixel2 = self.criterion_pixel(render2_flipped, rec2_flipped) * self.opt.lambda_pixel
        self.loss_pixel3 = self.criterion_pixel(self.render1, rec2_flipped) * self.opt.lambda_pixel
        self.loss_pixel4 = self.criterion_pixel(render2_flipped, self.rec1) * self.opt.lambda_pixel

        self.loss_all = (self.loss_pixel1 + self.loss_pixel2 + self.loss_pixel3 + self.loss_pixel4) / 4
        # self.loss_all = self.loss_pixel1
        # self.loss_all = (self.loss_pixel1 + self.loss_pixel2) / 2
        loss = self.loss_all
        loss.backward()
        
        # # 梯度裁剪
        # torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=1.0)
        
        self.optimizer.step()
