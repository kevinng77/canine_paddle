import abc
import os
from random import random

import h5py
import time
from paddle.amp import GradScaler, auto_cast
import paddlenlp
from tydi_canine_model.modeling import CanineForTydiQA
from tools.data_utils import get_dataloader
import logging
import random
import numpy as np
from paddle import distributed as dist
from tools.training_utils import *

logger = logging.getLogger(__name__)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    # Arguments for IO path
    parser.add_argument("--output_dir",
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--input_dir",
                        help="Precomputed input ids for training.")
    parser.add_argument("--state_dict_path",
                        help="path for trained model weight.")
    parser.add_argument("--checkout_steps", type=int, default=40000, help="number of steps for save checkout point")


    # Arguments for tydi evaluation
    parser.add_argument("--max_seq_length", type=int,default=2048,
                        help="The maximum total input sequence length after WordPiece tokenization. "
                             "Sequences longer than this will be truncated, and sequences shorter "
                             "than this will be padded.")

    # Arguments for Fine-tuning
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=450000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_proportion", type=float, default=0.1,
                        help="Proportion of steps for linear warmup, 0.1 for 10%.")
    parser.add_argument("--logging_steps", type=int,default=20)
    parser.add_argument("--scale_loss", type=float, default=4096, help="The value of scale_loss for fp16.", )
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--dev_split_ratio", type=float, default=0.02)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()
    return args


class CanineRunner(metaclass=abc.ABCMeta):
    """See run_tydi.py. All attributes are copied directly from the args."""

    def __init__(self, args):
        self.output_dir = args.output_dir
        self.input_dir = args.input_dir
        self.state_dict_path = args.state_dict_path
        self.checkout_steps = args.checkout_steps
        self.max_seq_length = args.max_seq_length

        self.max_steps = args.max_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.warmup_proportion = args.warmup_proportion
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.logging_steps = args.logging_steps
        self.scale_loss = args.scale_loss
        self.seed = args.seed
        self.dev_split_ratio = args.dev_split_ratio
        self.fp16 = args.fp16

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger_init()
        self.save_weight_file_name = f"tydi_seed_{self.seed}.pdparams"

    def split_training_samples(self, shuffle=False):
        """
        Indeed, paddle dataloader and sampler provided shuffle and index, but
        this is just in order to split the train and dev set from h5df database.

        Returns:
             List[int],List[int]: sample ids for train samples and dev samples.
        """
        with h5py.File(self.input_dir, 'r') as fp:
            num_samples = fp[feature_group_name].len()
        sample_ids = list(range(num_samples))
        if shuffle:
            random.shuffle(sample_ids)
        split = int(num_samples * self.dev_split_ratio)

        return sample_ids[split:], sample_ids[:split]

    def save_ckpt(self, model, name="tydi.pdparams"):
        if dist.get_rank() == 0:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            paddle.save(model.state_dict(), os.path.join(self.output_dir, name))

    def logger_init(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    os.path.join(self.output_dir, f"canine_training_{self.seed}.log"),
                    mode="w",
                    encoding="utf-8", ),
                logging.StreamHandler()
            ], )
        logger.info("**********  Configuration Arguments **********")
        for arg, value in sorted(vars(self).items()):
            logger.info(f"{arg}: {value}")
        logger.info("**************************************************")

    def load_model(self):
        model = CanineForTydiQA.from_pretrained('canine-s')
        if self.max_seq_length > model.canine.max_position_embeddings:
            raise ValueError(
                f"Canine only supports seq length <= {model.canine.max_position_embeddings}")

        if self.state_dict_path is not None:
            if os.path.isdir(self.state_dict_path):
                logger.info(f"\n>>>>>> loading weight from {self.state_dict_path}")
                load_state_dict = paddle.load(self.state_dict_path)
                model.set_state_dict(load_state_dict)

        if dist.get_world_size() > 1:
            model = paddle.DataParallel(model)

        return model

    def train(self):
        """
        Please refer to `note.md` at root directory for more information about hyperparameter and training
        settings.
        """
        if dist.get_world_size() > 1:
            dist.init_parallel_env()
        random.seed(self.seed)
        np.random.seed(self.seed)
        paddle.seed(self.seed)

        model = self.load_model()

        train_samples_ids, dev_samples_ids = self.split_training_samples()

        train_data_loader = get_dataloader(sample_ids=train_samples_ids,
                                           h5df_path=self.input_dir,
                                           batch_size=self.batch_size,
                                           is_train=True)

        dev_data_loader = get_dataloader(sample_ids=dev_samples_ids,
                                         h5df_path=self.input_dir,
                                         batch_size=self.batch_size,
                                         is_train=True)

        num_train_steps = min(int(len(train_data_loader) // self.gradient_accumulation_steps *
                              self.epochs), self.max_steps)

        logger.info(">>> num_training_samples: %d", len(train_samples_ids))
        logger.info(">>> num_dev_samples: %d", len(dev_samples_ids))
        logger.info(">>> theory batch size (batch * gradient step * n_gpus): %d",
                    self.batch_size * self.gradient_accumulation_steps * dist.get_world_size())
        logger.info(">>> num_train_steps for each GPU: %d", num_train_steps)

        lr_scheduler = paddlenlp.transformers.LinearDecayWithWarmup(
            learning_rate=self.learning_rate,
            total_steps=num_train_steps,
            warmup=int(num_train_steps * self.warmup_proportion))

        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            epsilon=1e-6,
            apply_decay_param_fun=lambda x: x in decay_params,
        )
        criterion = CrossEntropyLossForTydi()

        scaler = None
        if self.fp16:
            scaler = GradScaler(init_loss_scaling=self.scale_loss)

        model.train()
        global_step = 0
        time1 = time.time()
        optimizer.clear_grad()
        losses = paddle.to_tensor([0.0])
        loss_list, dev_loss_list, acc_list, diff_list = [], [], [], []  # for multi-gpu
        logger.info(">>> start training...")
        for epoch in range(1, int(self.epochs) + 1):
            for step, batch in enumerate(train_data_loader, start=1):
                with auto_cast(enable=self.fp16,
                               custom_white_list=["softmax", "gelu"]):
                    logits = model(input_ids=batch['input_ids'],
                                   token_type_ids=batch['segment_ids'],
                                   attention_mask=batch['input_mask']
                                   )

                    loss = criterion(logits=logits,
                                     start_positions=batch['start_positions'],
                                     end_positions=batch['end_positions'],
                                     answer_types=batch['answer_types'])

                    loss = loss / self.gradient_accumulation_steps
                    losses += loss.detach()  # losses for logging only

                if self.fp16:
                    scaled = scaler.scale(loss)
                    scaled.backward()
                else:
                    loss.backward()

                if step % self.gradient_accumulation_steps == 0:
                    global_step += 1
                    if global_step > self.max_steps:
                        break
                    if self.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.clear_grad()
                    lr_scheduler.step()

                    if global_step % self.logging_steps == 0 or \
                            global_step == num_train_steps:

                        if dist.get_world_size() > 1:
                            dist.barrier()
                        local_loss = losses / self.logging_steps
                        dev_loss_tensor, acc, diff = self.evaluate(model=model,
                                                                   dev_data_loader=dev_data_loader,
                                                                   criterion=criterion)
                        if dist.get_world_size() > 1:
                            dist.all_gather(loss_list, local_loss)
                            dist.all_gather(dev_loss_list, dev_loss_tensor)
                            dist.all_gather(acc_list, acc)
                            dist.all_gather(diff_list, diff)

                            if dist.get_rank() == 0:
                                logging_loss = (paddle.stack(loss_list).sum() / len(
                                    loss_list)).item()
                                dev_loss = (paddle.stack(dev_loss_list).sum() / len(
                                    dev_loss_list)).item()
                                logging_acc = (paddle.stack(acc_list).sum() / len(
                                    acc_list)).item()
                                logging_diff = (paddle.stack(diff_list).sum() / len(
                                    diff_list)).item()

                                logger.info(f"Step {global_step}/{num_train_steps} train loss {logging_loss:.4f}"
                                            f" dev loss {dev_loss:.4f} acc {logging_acc:.2f}% diff {logging_diff:.2f}"
                                            f" time {(time.time() - time1) / 60:.2f}min"
                                            )
                            dist.barrier()
                        else:
                            logging_loss = local_loss.item()
                            logger.info(f"Step {global_step}/{num_train_steps} train loss {logging_loss:.4f}"
                                        f" dev loss {dev_loss_tensor.item():.4f} acc {acc.item():.2f}% "
                                        f"diff {diff.item():.2f}"
                                        f" time {(time.time() - time1) / 60:.2f}min"
                                        )
                        losses = paddle.to_tensor([0.0])
                        time1 = time.time()
                        model.train()
                    if global_step % self.checkout_steps == 0:
                        self.save_ckpt(model, name=f"{global_step}_{self.seed}_tydi.pdparams")

        logger.info(f"training done, total steps trained: {global_step}")
        self.save_ckpt(model, name=self.save_weight_file_name)

    @paddle.no_grad()
    def evaluate(self, model, dev_data_loader, criterion):
        """
        This Evaluating step is just for observing the trend of fine-tuning and debugging. It does not
        provide any information about the final Tydi task testing score.
        Returns:
            losses (Tensor): losses on development set.
            total_acc (Tensor): Mean Accuracy on Answer type prediction task.
            total_diff (Tensor): difference between target answer span and predicted answer span,
                           w.r.t. the start and end index.
        """
        model.eval()
        total_acc, total_diff = paddle.to_tensor([0.0]), paddle.to_tensor([0.0])
        losses = paddle.to_tensor([0.0])
        acc_count, diff_count = 0, 0
        for step, batch in enumerate(dev_data_loader, start=1):
            start_logits, end_logits, type_logits = model(input_ids=batch['input_ids'],
                                                          token_type_ids=batch['segment_ids'],
                                                          attention_mask=batch['input_mask']
                                                          )
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']
            answer_types = batch['answer_types']
            loss = criterion(logits=(start_logits, end_logits, type_logits),
                             start_positions=start_positions,
                             end_positions=end_positions,
                             answer_types=answer_types)
            start_pred = paddle.argmax(start_logits, axis=-1)
            end_pred = paddle.argmax(end_logits, axis=-1)
            type_pred = paddle.argmax(type_logits, axis=-1)

            total_acc += paddle.sum((type_pred == answer_types)[answer_types != 0])
            acc_count += paddle.sum(answer_types != 0).item()

            start_mask = start_positions != 0
            total_diff += paddle.sum(paddle.abs(start_pred - start_positions)[start_mask])
            total_diff += paddle.sum(paddle.abs(end_pred - end_positions)[start_mask])
            diff_count += paddle.sum(start_mask).item()

            losses += loss.detach()

        losses /= len(dev_data_loader) + 1
        total_acc /= acc_count + 1
        total_diff /= diff_count * 2 + 1
        return losses, total_acc * 100, total_diff

if __name__ == "__main__":
    args = get_args()
    runner = CanineRunner(args)
    runner.train()
