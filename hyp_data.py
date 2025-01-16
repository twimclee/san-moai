class MHyp:

    def __init__(self):
        self.cfg = 'stylegan3-r'
        self.imgsize = 512
        self.gpus = 1
        self.batch = 4
        self.batch_gpu = 1
        self.kimg = 1000
        self.workers = 0
        self.snap = 50
        self.syn_layers = 14

        self.mirror = False

    def print_data(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

class MData:

    def __init__(self):
        self.names = None

    def print_data(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

