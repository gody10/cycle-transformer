from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behaviour during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--subtraction_eval', action='store_true',
                            help='Evaluate subtraction images: compare (source - generated) vs (source - ground_truth)')
        parser.add_argument('--save_subtractions', action='store_true',
                            help='Save subtraction NIfTI volumes (source - generated, source - ground_truth) denormalized to HU')
        parser.add_argument('--roi_eval', action='store_true',
                            help='Compute vessel ROI metrics: Dice, CNR, vessel-masked MAE/PSNR. '
                                 'Uses thresholded GT subtraction to define vessel regions.')
        parser.add_argument('--vessel_threshold_hu', type=float, default=50.0,
                            help='HU threshold for vessel detection in subtraction images (default 50). '
                                 'Voxels with subtraction > threshold are considered vessels.')
        parser.add_argument('--fid_eval', action='store_true',
                            help='Compute 2D FID using the centre axial slice of each volume with InceptionV3 features.')
        parser.add_argument('--frd_eval', action='store_true',
                            help='Compute FRD (Fréchet Radiomic Distance) using radiomic features '
                                 'extracted from axial slices of each volume.')
        parser.add_argument('--frd_num_slices', type=int, default=5,
                            help='Number of evenly-spaced axial slices per volume for FRD (default: 5).')
        parser.add_argument('--save_generated_dir', type=str, default=None,
                            help='Directory to save generated volumes as NIfTI ({patient_id}_pred.nii.gz) '
                                 'in [0,1] space after inference.')
        parser.add_argument('--load_generated_dir', type=str, default=None,
                            help='Directory to load pre-generated NIfTI volumes from, skipping model inference. '
                                 'Files must be named {patient_id}_pred.nii.gz in [0,1] space.')
        # CyTran-specific architecture options (must match training)
        parser.add_argument('--ngf_cytran', type=int, default=16, help='number of generator filters for CyTran')
        parser.add_argument('--n_downsampling', type=int, default=3, help='number of downsampling layers')
        parser.add_argument('--depth', type=int, default=3, help='transformer depth')
        parser.add_argument('--heads', type=int, default=6, help='number of attention heads')
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')
        # rewrite default values
        parser.set_defaults(model='test')
        self.isTrain = False
        return parser
