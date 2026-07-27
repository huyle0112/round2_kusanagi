#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0
        # Optional deterministic holdout from the real COLMAP images. This is
        # independent of the competition test_poses.csv and is intended for
        # validation/ablation runs with ground truth.
        self.validation_ratio = 0.0
        self.validation_seed = 42
        # When validation_ratio is zero, metrics can still be monitored on a
        # deterministic sample of training cameras without withholding them.
        # A value of zero disables this proxy-validation mode.
        self.validation_sample_count = 0
        # Undistort SIMPLE_RADIAL inputs for the pinhole rasterizer and warp
        # renders back to the raw camera domain when exporting.
        self.correct_radial_distortion = False

        # Per-camera appearance embeddings are useful for uncontrolled exposure,
        # but test-pose UIDs do not identify matching train cameras in this dataset.
        # Keep them disabled by default so colour is predicted from geometry/view only.
        self.appearance_dim = 0
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002

        # GaussianPro is the only anchor-growth path when enabled. It propagates
        # multi-view geometry throughout training while Scaffold-GS continues
        # to optimize and prune the resulting anchors.
        self.use_gaussianpro = False
        self.gaussianpro_start_iter = 3000
        self.gaussianpro_add_until_iter = 15_000
        self.gaussianpro_refine_until_iter = 24_000
        self.gaussianpro_interval = 50
        self.gaussianpro_neighbors = 4
        self.gaussianpro_graph_samples = 4096
        self.gaussianpro_min_overlap = 0.05
        self.gaussianpro_downsample = 4
        self.gaussianpro_patch_radius = 2
        self.gaussianpro_patchmatch_iterations = 3
        self.gaussianpro_opacity_threshold = 0.5
        self.gaussianpro_coverage_threshold = 0.5
        self.gaussianpro_min_consistent_views = 3
        self.gaussianpro_max_photo_error = 0.25
        self.gaussianpro_reprojection_threshold = 2.0
        self.gaussianpro_depth_consistency_threshold = 0.03
        self.gaussianpro_normal_consistency_threshold = 0.5
        self.gaussianpro_depth_discrepancy_threshold = 0.20
        self.gaussianpro_max_anchors_per_step = 128
        self.gaussianpro_voxel_factor = 1.0
        self.gaussianpro_seed = 42
        self.gaussianpro_max_anchor_multiplier = 1.25
        self.gaussianpro_refine_radius_factor = 1.5
        self.gaussianpro_refine_rate = 0.1
        self.gaussianpro_confidence_decay = 0.995
        self.gaussianpro_prune_interval = 500
        self.gaussianpro_prune_confidence = 0.25
        self.gaussianpro_prune_opacity = 0.01
        self.gaussianpro_prune_grace_iters = 1000

        # Geometry supervision stops at refine_until. Flatness is a hinge
        # target, so ratios below the target are not pushed towards zero.
        self.gaussianpro_flatness_target = 0.20
        self.gaussianpro_flatness_floor = 0.02
        self.lambda_gaussianpro_flatness = 0.001
        self.lambda_gaussianpro_normal_l1 = 0.001
        self.lambda_gaussianpro_normal_cos = 0.001
        self.lambda_gaussianpro_feature_l1 = 0.0002
        self.lambda_gaussianpro_feature_cos = 0.0002
        self.gaussianpro_feature_interval = 4
        self.gaussianpro_feature_samples = 512

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
