from .base_options import BaseOptions


class AugOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # training parameters
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        self.isTrain = True
        return parser
