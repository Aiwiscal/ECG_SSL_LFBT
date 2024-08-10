class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]

    def __call__(self, sample):
        output = [transform(sample) for transform in self.transforms]
        return output
