


class ActiveLearner():

    def __init__(self):
        raise NotImplementedError

    def fit(self, dataloader):
        raise NotImplementedError

    def score(self, dataloader):
        raise NotImplementedError

    def inference(self, dataset):
        raise NotImplementedError

    def query(self, dataset, algorithm, query_size):
        inference_object = self.inference(dataset)
        return algorithm(inference_object, query_size)