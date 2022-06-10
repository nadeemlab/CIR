import logging
import wandb

logger = logging.getLogger(__name__)


class Trainer(object):
    def training_step(self, data, epoch):
        # Get the minibatch

        self.optimizer.zero_grad()
        loss, log = self.net.loss(data, epoch)
        loss.backward()
        self.optimizer.step()
        # embed()

        return log

    def __init__(self, net, trainloader, optimizer, save_path, evaluator, config):

        self.net = net
        self.trainloader = trainloader
        self.optimizer = optimizer
        
        self.numb_of_itrs = config.numb_of_itrs
        self.eval_every = config.eval_every
        self.save_path = save_path

        self.evaluator = evaluator
        self.config = config

    def train(self, start_iteration=1):

        print("Start training...")

        self.net = self.net.train()
        iteration = start_iteration

        print_every = 1
        for epoch in range(10000000):  # loop over the dataset multiple times

            for itr, data in enumerate(self.trainloader):

                # training step
                loss = self.training_step(data, start_iteration)

                if iteration % print_every == 0:
                    log_vals = {}
                    for key, value in loss.items():
                        log_vals[key] = value / print_every
                    log_vals['iteration'] = iteration

                if self.config.wab:
                    wandb.log(log_vals)

                iteration = iteration + 1

                if iteration % self.eval_every == self.eval_every-1:  # print every K epochs
                    self.evaluator.evaluate(int((iteration+1)/self.eval_every))
                    self.net = self.net.train()
                if iteration > self.numb_of_itrs:
                    break

        logger.info("... end training!")
