from .libs import *

class Options():
    def __init__(self, notebook):
        self.initialized = False
        self.notebook = notebook
        self.glr = 1e-5
        # self.glr = PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5]),
        self.dlr = 1e-7
        # self.dlr = PiecewiseConstantDecay(boundaries=[100000], values=[1e-6, 1e-7]),
        self.Lambda = 1e-5

    def initialize(self, parser):
        parser.add_argument('--glr', type=float, default=1e-5, help='gen learning rate')
        parser.add_argument('--dlr', type=float, default=1e-7, help='disc learning rate')
        parser.add_argument('--Lambda', type=float, default=1e-5, help='lambda')
        self.initialized = True
        return parser
    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser

        return parser.parse_args() if not self.notebook else self

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):

        opt = self.gather_options()
        self.opt = opt
        return self.opt
notebook = True
opt = Options(notebook) if not notebook else Options(True).parse()

def __main__():
    pass
if __name__ == "__main__":
    pass