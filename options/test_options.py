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
