import logging
import wandb
from tqdm import tqdm

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

        self.numb_of_epochs = config.numb_of_epochs
        self.save_path = save_path

        self.evaluator = evaluator
        self.config = config

    def train(self, start_epoch=1):

        print("Start training...")

        self.net = self.net.train()
        print_every = 10
        log_vals = {}
        dataset_name = self.evaluator.data["training"].name
        prefix = dataset_name + '/' + "training" + '/'
        for epoch in range(start_epoch, 10000):  # loop over the dataset multiple times
            with tqdm(self.trainloader, unit="sample") as tepoch:
                for itr, data in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    iteration = (epoch - 1) * len(self.trainloader) + itr
                    
                    # training step
                    loss = self.training_step(data, start_epoch)

                    if iteration % print_every == print_every-1:
                        for key, value in log_vals.items():
                            log_vals[key] = value / print_every
                        log_vals['iteration'] = iteration
                        if self.config.wab:
                            wandb.log(log_vals)
                        tepoch.set_postfix(loss=log_vals[prefix + 'loss'].cpu().numpy())
                        log_vals = {}
                    else:
                        for key, value in loss.items():
                            try:
                                log_vals[prefix + key] += value
                            except:
                                log_vals[prefix + key] = value

                self.evaluator.evaluate(epoch)
                self.net = self.net.train()

                if epoch >= self.numb_of_epochs:
                    break

        logger.info("... end training!")
        wandb.finish()
