import mmcv

from models.base_bev_depth import BaseBEVDepth
from layers.backbones.rvt_lss_fpn import RVTLSSFPN
from layers.backbones.pts_backbone import PtsBackbone

logger = mmcv.utils.get_logger('mmdet')
logger.setLevel('WARNING')

__all__ = ['CameraRadarNetDet']


class CameraRadarNetDetDepth(BaseBEVDepth):
    """Source code of `CRN`, `https://arxiv.org/abs/2304.00670`.

    Args:
        backbone_img_conf (dict): Config of image backbone.
        backbone_pts_conf (dict): Config of point backbone.
        fuser_conf (dict): Config of BEV feature fuser.
        head_conf (dict): Config of head.
    """

    def __init__(self, backbone_img_conf, backbone_pts_conf, fuser_conf, head_conf):
        super(BaseBEVDepth, self).__init__()
        self.backbone_img = RVTLSSFPN(**backbone_img_conf)
        self.backbone_pts = PtsBackbone(**backbone_pts_conf)

        self.radar_view_transform = backbone_img_conf['radar_view_transform']

        # inference time measurement
        self.idx = 0
        self.times_dict = {
            'img': [],
            'img_backbone': [],
            'img_dep': [],
            'img_transform': [],
            'img_pool': [],

            'pts': [],
            'pts_voxelize': [],
            'pts_backbone': [],
            'pts_head': [],

            'fusion': [],
            'fusion_pre': [],
            'fusion_layer': [],
            'fusion_post': [],

            'head': [],
            'head_backbone': [],
            'head_head': [],
        }

    def forward(self,
                sweep_imgs,
                mats_dict,
                sweep_ptss=None,
                is_train=False
                ):
        """Forward function for BEVDepth

        Args:
            sweep_imgs (Tensor): Input images.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            sweep_ptss (Tensor): Input points.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """

        ptss_context, ptss_occupancy, _ = self.backbone_pts(sweep_ptss)
        _, depth, _ = self.backbone_img(sweep_imgs,
                                        mats_dict,
                                        ptss_context,
                                        ptss_occupancy,
                                        return_depth=True)
        return depth
