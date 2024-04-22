from .base_options import BaseOptions

class TestOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = False
        # self.planner = 'geometry-drop'

        self.parser.add_argument('--results_dir', type=str, default='./logs', help='saves results here.')
        self.parser.add_argument('--total_iters', type=int, default=10, help='# of iter for testing')
        self.parser.add_argument('--planner', type=str, default='GeometryDropPlanner', help='inference planner')

        # for diversity test
        self.parser.add_argument('--test_diversity', action='store_true', help='test diversity')
        self.parser.add_argument('--testdir', type=str, default='', help='test directory prefix')

        # use bbox mesh to initialize
        self.parser.add_argument('--use_bbox_mesh', action='store_true', help='use bbox mesh to initialize')

        self.phase = 'test'
        # self.opt.results_dir = self.opt.logs_dir