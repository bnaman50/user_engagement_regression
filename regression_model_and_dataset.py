import sys

import pytorch_lightning as pl
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import math
import traceback
from nested_lookup import nested_lookup
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
import settings

## TODO:
'''
Most of the time is spent in data loading based on the profiler. 
One of the downsides of this data-loader is that I need to specify the training samples beforehand. 
In future, I would like to implement IterableDataset along with BufferedShuffleDataset for train data 
shuffling to certain extent.
'''

# Create Dataset
class CSVDataset(Dataset):
    def __init__(self, path, tokenizer, nb_samples):
        """
        :param path: Path to the CSV file
        :param tokenizer: tokenizer
        :param nb_samples: total samples in the corresponding file
        """
        super().__init__()
        self.path = path
        self.chunksize = 1
        self.len = nb_samples // self.chunksize
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        x = next(
            pd.read_csv(
                self.path,
                skiprows=range(1, index * self.chunksize + 1),  #+1, since we skip the header
                chunksize=self.chunksize,
                ))
        title = str(x['title'].values[0]).strip()
        assert isinstance(title, str), 'Something is wrong with titles'
        up_votes = float(x['up_votes'].values[0])
        assert isinstance(up_votes, float), 'Something is wrong with up_votes'
        return {'titles': title, 'up_votes': up_votes}

    def __len__(self):
        return self.len

    def collate_fn(self, batch):
        titles = [x["titles"] for x in batch]
        up_votes = [x["up_votes"] for x in batch]
        # titles, up_votes = zip(*batch) ## unzips the batch because of the * operator

        try:
        ## Working
            batch_encoding = self.tokenizer(titles,
                                            add_special_tokens=True,
                                            pad_to_max_length=True,
                                            padding='longest',
                                            truncation=True,
                                            return_tensors='pt',
                                            ).data
        except Exception as e:
            print(e)
            print(traceback.print_exc())
            print('I was facing some weird error. Realized that I am running out of RAM becaus of very high num-workers')
            print('If you also see it, reduce you number of workers')
            sys.exit(0)

        batch_encoding["labels"] = torch.tensor([x for x in up_votes])
        return batch_encoding


class BertForRegression(pl.LightningModule):
    def __init__(self,
                 model_name='distilbert-base-cased',
                 learning_rate=5e-5,
                 train_batch_size=32,
                 val_batch_size=32,
                 test_batch_size=32,
                 freeze_encoder=True,
                 num_workers=4,
                 pin_memory=True,
                 num_labels=1,
                 verbose=True,
                 **kwargs):
        super(BertForRegression, self).__init__()

        ## that's how tuner wants it
        self.learning_rate = learning_rate

        self.save_hyperparameters()  # Now possible to access parameters from hparams

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        if 'distil' in model_name:
            model_name_key = 'distilbert'
        else:
            model_name_key = 'bert'

        if freeze_encoder:
            for param in eval(f'self.model.{model_name_key}').parameters():
                param.requires_grad = False

        ## Check whether model freezing happened or not. Working
        # self.model.train()
        # train_params = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]

        ## for accumulated grads
        ## TODO: this MIGHT chnage in early stopping
        self.accumulated_train_loss_per_step = []
        self.accumulated_train_loss_counter = 0

    @staticmethod
    def add_model_and_data_specific_args(parent_parser, root_dir):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name',
                            help='Model name to be fine-tuned [Options = distilbert-base-cased, distilbert-base-uncased, '
                                 'bert-base-cased, bert-base-uncased]'
                                 '(Default - distilbert-base-cased)',
                            default='distilbert-base-cased')
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.") ## Not being used RN
        parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
        parser.add_argument('--pin_memory', action='store_true', help='Whether to pin memory or not. Default=False')
        parser.add_argument("--num_labels", default=1, type=int, help=f"Number of output labels. "
                                                                      f"Helpful when we want to predict both up_votes and down_votes")
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--val_batch_size", default=32, type=int)
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
        parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float,
                            help="Max gradient norm")
        parser.add_argument(
            "--gradient_accumulation_steps",
            dest="accumulate_grad_batches",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
        parser.add_argument('--verbose', action='store_true', help='Wthether to print the training loss or not. Default = False')
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

        return parser

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def _step(self, batch):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        ## Write to the logger (Default is tensorboard)
        # len_train_loader = len(self.train_dataloader())
        x_axis = self.current_epoch * self.len_train_loader + batch_idx  ## limit_prac logic is already considered
        self.logger.experiment.add_scalar("Loss/train_loss_per_step", loss, x_axis)
        self.logger.experiment.add_scalar("my_epoch_plot", self.current_epoch, x_axis)

        if self.hparams.verbose:
            print(f'Type: Train, Epoch: {self.current_epoch: 2d}/{self.trainer.max_epochs}, '
                  f'BatchIdx: {batch_idx: 3d}/{self.len_train_loader}, '
                  f'Train_Loss: {loss: 8.3f}')

        ## Log the avergae accumulated train loss
        if self.hparams.accumulate_grad_batches > 1:
            self.accumulated_train_loss_per_step.append(loss.detach())
            if len(self.accumulated_train_loss_per_step) >= self.hparams.accumulate_grad_batches:
                avg_acc_loss = torch.stack(self.accumulated_train_loss_per_step).mean()
                self.logger.experiment.add_scalar("Loss/Accumulated_Train_Loss", avg_acc_loss,
                                                  self.accumulated_train_loss_counter)
                self.accumulated_train_loss_per_step = []
                self.accumulated_train_loss_counter += 1

        return {"loss": loss, "x_axis": x_axis, "actual_samples": batch['labels'].numel()}


    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        if self.hparams.verbose:
            print(f'For epoch {self.current_epoch: 2d}, Avg Train Loss is {avg_train_loss: 8.3f}\n')

        self.logger.experiment.add_scalar("Loss/Avg_Train_Loss",
                                          avg_train_loss,
                                          self.current_epoch)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        if self.hparams.verbose:
            print(f'Type: Val, Epoch: {self.current_epoch: 2d}/{self.trainer.max_epochs}, '
                  f'BatchIdx: {batch_idx: 3d}/{self.len_val_loader}, '
                  f'Per_Step_Val_Loss: {loss: 8.3f}')

        ## Write to the logger
        # len_val_loader = len(self.val_dataloader())
        x_axis = self.current_epoch * self.len_val_loader + batch_idx
        self.logger.experiment.add_scalar("Loss/Val_loss_per_step", loss, x_axis)

        val_loss_per_step = 'val_loss_per_step'
        return {val_loss_per_step: loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss_per_step'] for x in outputs]).mean()
        ## This is required for check-pointing
        self.log('val_loss', avg_loss)

        self.logger.experiment.add_scalar("Loss/Avg_Val_Loss", avg_loss, self.current_epoch)

        if self.hparams.verbose:
            print(f'Type: Validation, Epoch: {self.current_epoch: 2d}/{self.trainer.max_epochs}, '
                  f'AvgValLoss: {avg_loss: 8.3f}\n')

    def get_loader_length(self, inp_loader, limit_frac):
        if limit_frac == 1.0:
            out_len = len(inp_loader)
        else:
            out_len = math.floor(len(inp_loader)*limit_frac)

        return  out_len

    def setup(self, mode): ## first setup is called to get the data
        if mode == "test":
            self.dataset_size = len(self.test_dataloader().dataset)*self.trainer.limit_test_batches
        else:
            # self.train_loader = self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True) ## I don't think this is required here

            ## for scheduler purposes
            self.dataset_size = len(self.train_dataloader().dataset)*self.trainer.limit_train_batches

        self.len_train_loader = self.get_loader_length(self.train_dataloader(), self.trainer.limit_train_batches)
        self.len_val_loader = self.get_loader_length(self.val_dataloader(), self.trainer.limit_val_batches)
        aa = 1

    def total_steps(self):
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        effective_steps_per_epoch = math.ceil((self.dataset_size / effective_batch_size))
        return effective_steps_per_epoch * self.hparams.max_epochs

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate,
                          eps=1e-8, correct_bias=False)

        ## LR Scheduler ##TODO: need to check this properly
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps()
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def get_dataset(self, type_path):
        if type_path == 'train':
            nb_samples = settings.n_train_samples
        elif type_path == 'val':
            nb_samples = settings.n_val_samples
        elif type_path == 'test':
            nb_samples = settings.n_test_samples
        else:
            print(f'Not yet implementted.')
            sys.exit(0)

        dataset = CSVDataset(path=os.path.join(settings.data_dir, f'{type_path}.csv'),
                             tokenizer=self.tokenizer,
                             nb_samples=nb_samples,
                             )
        return dataset

    def get_dataloader(self, type_path, batch_size, shuffle=False, num_workers=0, pin_memory=False):
        dataset = self.get_dataset(type_path)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def train_dataloader(self):
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size,
                                         shuffle=True, num_workers=self.hparams.num_workers,
                                         pin_memory=self.hparams.pin_memory)
        return dataloader

    def val_dataloader(self):
        return self.get_dataloader("val", batch_size=self.hparams.val_batch_size,
                                   num_workers=self.hparams.num_workers,
                                   pin_memory=self.hparams.pin_memory)

    def test_dataloader(self):
        return self.get_dataloader("test", batch_size=self.hparams.test_batch_size,
                                   num_workers=self.hparams.num_workers,
                                   pin_memory=self.hparams.pin_memory)

class LoggingCallback(pl.Callback):

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when the train epoch ends."""
        if pl_module.hparams.verbose:
            print(f'Start of validation epoch')

    def on_train_end(self, trainer, pl_module):
        if pl_module.hparams.verbose:
            print('Training done!\n')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        lrs = {f"lr_group_{i}": param["lr"] for i, param in enumerate(pl_module.trainer.optimizers[0].param_groups)}

        for key, val in lrs.items():
            x_axis = nested_lookup('x_axis', outputs)
            assert len(x_axis), 'training_step must return value of x_axis for logging'
            pl_module.logger.experiment.add_scalar(f"LR_Scheduler/{key}", val, x_axis[0])
