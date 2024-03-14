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
        self.parser.add_argument('--print_collision_loss', action='store_true', 
                                                        help='print collision loss for choosing the best model')

        # test a single model
        self.parser.add_argument('--model_id', default=None, type=str, help='model id to optimize')

        self.phase = 'test'
        # self.opt.results_dir = self.opt.logs_dir