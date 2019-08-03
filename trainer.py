class Trainer:
    def __init__(self, n_iterations, n_log_iterations):
        self.n_iterations = n_iterations
        self.n_log_iterations = n_log_iterations

    def train(self, model, dataset):
        """
        :param dataset: dataset.__call__ returns batch
        :param model: requires model.iteration, model.metrics, model.train_step(batch)
        """
        for batch in dataset:
            i = model.iteration.numpy()
            model.train_step(batch)
            string = f'\riteration: {i + 1}'
            for name, metric in model.metrics.items():
                string += f', {name}: {metric.result().numpy():.5e}'
                if i % self.n_log_iterations == self.n_log_iterations - 1:
                    metric.reset_states()
            print(string, end='')
            if i % self.n_log_iterations == self.n_log_iterations - 1:
                print()
            if i == self.n_iterations - 1:
                break
