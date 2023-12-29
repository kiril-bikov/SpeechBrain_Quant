#!/usr/bin/env/python3
"""Recipe for training a sequence-to-sequence ASR system with librispeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beamsearch coupled with a neural
language model.

To run this recipe, do the following:
> python train.py hparams/train_BPE1000.yaml

With the default hyperparameters, the system employs a CRDNN encoder.
The decoder is based on a standard  GRU. Beamsearch coupled with a RNN
language model is used  on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split (e.g, train-clean 100 rather than the full one), and many
other possible variations.

This recipe assumes that the tokenizer and the LM are already trained.
To avoid token mismatches, the tokenizer used for the acoustic model is
the same use for the LM.  The recipe downloads the pre-trained tokenizer
and LM.

If you would like to train a full system from scratch do the following:
1- Train a tokenizer (see ../../Tokenizer)
2- Train a language model (see ../../LM)
3- Train the acoustic model (with this code).



Authors
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
 * Andreas Nautsch 2021
"""

import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path

import torch
import torch.utils.data
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization import tensor_quant
from pytorch_quantization.tensor_quant import QuantDescriptor

from tqdm import tqdm

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        x = self.modules.enc(feats.detach())
        e_in = self.modules.emb(tokens_bos)  # y_in bos + tokens
        h, _ = self.modules.dec(e_in, x, wav_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                # Output layer for ctc log-probabilities
                logits = self.modules.ctc_lin(x)
                p_ctc = self.hparams.log_softmax(logits)
                return p_ctc, p_seq, wav_lens
            else:
                return p_seq, wav_lens
        else:
            if stage == sb.Stage.VALID:
                p_tokens, scores = self.hparams.valid_search(x, wav_lens)
            else:
                p_tokens, scores = self.hparams.test_search(x, wav_lens)
            return p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        current_epoch = self.hparams.epoch_counter.current
        if stage == sb.Stage.TRAIN:
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                p_ctc, p_seq, wav_lens = predictions
            else:
                p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # Add ctc loss if necessary
        if (
            stage == sb.Stage.TRAIN
            and current_epoch <= self.hparams.number_of_ctc_epochs
        ):
            loss_ctc = self.hparams.ctc_cost(
                p_ctc, tokens, wav_lens, tokens_lens
            )
            loss = self.hparams.ctc_weight * loss_ctc
            loss += (1 - self.hparams.ctc_weight) * loss_seq
        else:
            loss = loss_seq

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)
                    
    def quant_evaluate(
            self,
            test_set,
            train_set,
            max_key=None,
            min_key=None,
            progressbar=None,
            test_loader_kwargs={},
            train_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
                ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
                isinstance(test_set, torch.utils.data.DataLoader)
                or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
        """
        self.on_evaluate_start(max_key=max_key, min_key=min_key)

        quant_dataloader = self.make_dataloader(
            train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
        )
        quantize(self, quant_dataloader)
        """
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                    test_set,
                    dynamic_ncols=True,
                    disable=not progressbar,
                    colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                loss = self.evaluate_batch(batch, stage=sb.Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            self.on_stage_end(sb.Stage.TEST, avg_test_loss, None)
        self.step = 0
        return avg_test_loss   
    
    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """
        if self.test_only:
            logger.info(
                "Test only mode, skipping training and validation stages."
            )
            return

        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()
        
        quantize(self, train_set)

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Iterate epochs
        for epoch in epoch_counter:
            self._fit_train(train_set=train_set, epoch=epoch, enable=enable)
            self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa
        from speechbrain.dataio.dataloader import SaveableDataLoader  # noqa
        from speechbrain.dataio.batch import PaddedBatch  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        hop_size = dynamic_hparams["feats_hop_size"]

        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"] * (1 / hop_size),
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"] * (1 / hop_size),
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        train_batch_sampler,
        valid_batch_sampler,
    )


class QuantEmb(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, quant_desc_weight=None):
        super(QuantEmb, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.init_quantizer(quant_desc_weight)

    def init_quantizer(self, quant_desc_weight):
        if quant_desc_weight is not None:
            self._weight_quantizer = quant_nn.modules.tensor_quantizer.TensorQuantizer(quant_desc_weight)
        else:
            self._weight_quantizer = None

    def forward(self, input):
        if self._weight_quantizer is not None:
            quant_weight = self._weight_quantizer(self.weight)
        else:
            quant_weight = self.weight

        return F.embedding(
            input, quant_weight, padding_idx=self.padding_idx, max_norm=self.max_norm,
            norm_type=self.norm_type, scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse
        )

    @property
    def weight_quantizer(self):
        return self._weight_quantizer


def QuantConv(conv_layer, quant_desc_input, quant_desc_weight):
    quantized_conv_layer = quant_nn.QuantConv2d(
        in_channels=conv_layer.in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        groups=conv_layer.groups,
        bias=conv_layer.bias is not None,
        padding_mode=conv_layer.padding_mode,
        quant_desc_input=quant_desc_input,
        quant_desc_weight=quant_desc_weight
    )

    quantized_conv_layer.weight = conv_layer.weight

    if conv_layer.bias is not None:
        quantized_conv_layer.bias = conv_layer.bias

    return quantized_conv_layer


def QuantLinear(linear_layer, quant_desc_input, quant_desc_weight):
    quantized_linear_layer = quant_nn.QuantLinear(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        bias=linear_layer.bias is not None,
        quant_desc_input=quant_desc_input,
        quant_desc_weight=quant_desc_weight
    )

    quantized_linear_layer.weight = linear_layer.weight

    if linear_layer.bias is not None:
        quantized_linear_layer.bias = linear_layer.bias

    return quantized_linear_layer

def QuantMaxPool2d(maxpool_layer, quant_desc_input):
    quantized_maxpool_layer = quant_nn.QuantMaxPool2d(
        kernel_size=maxpool_layer.kernel_size,
        stride=maxpool_layer.stride,
        padding=maxpool_layer.padding,
        dilation=maxpool_layer.dilation,
        return_indices=maxpool_layer.return_indices,
        ceil_mode=maxpool_layer.ceil_mode,
        quant_desc_input=quant_desc_input
    )

    return quantized_maxpool_layer

def QuantLSTM(lstm_layer, quant_desc_input, quant_desc_weight):
    quantized_lstm_layer = quant_nn.QuantLSTM(
        input_size=lstm_layer.input_size,
        hidden_size=lstm_layer.hidden_size,
        num_layers=lstm_layer.num_layers,
        bias=lstm_layer.bias,
        batch_first=lstm_layer.batch_first,
        dropout=lstm_layer.dropout,
        bidirectional=lstm_layer.bidirectional,
        quant_desc_input=quant_desc_input,
        quant_desc_weight=quant_desc_weight
    )

    quantized_lstm_layer.bias_hh_l0 = lstm_layer.bias_hh_l0
    quantized_lstm_layer.bias_hh_l0_reverse = lstm_layer.bias_hh_l0_reverse
    quantized_lstm_layer.bias_hh_l1 = lstm_layer.bias_hh_l1
    quantized_lstm_layer.bias_hh_l1_reverse = lstm_layer.bias_hh_l1_reverse
    quantized_lstm_layer.bias_hh_l2 = lstm_layer.bias_hh_l2
    quantized_lstm_layer.bias_hh_l2_reverse = lstm_layer.bias_hh_l2_reverse
    quantized_lstm_layer.bias_hh_l3 = lstm_layer.bias_hh_l3
    quantized_lstm_layer.bias_hh_l3_reverse = lstm_layer.bias_hh_l3_reverse

    quantized_lstm_layer.bias_ih_l0 = lstm_layer.bias_ih_l0
    quantized_lstm_layer.bias_ih_l0_reverse = lstm_layer.bias_ih_l0_reverse
    quantized_lstm_layer.bias_ih_l1 = lstm_layer.bias_ih_l1
    quantized_lstm_layer.bias_ih_l1_reverse = lstm_layer.bias_ih_l1_reverse
    quantized_lstm_layer.bias_ih_l2 = lstm_layer.bias_ih_l2
    quantized_lstm_layer.bias_ih_l2_reverse = lstm_layer.bias_ih_l2_reverse
    quantized_lstm_layer.bias_ih_l3 = lstm_layer.bias_ih_l3
    quantized_lstm_layer.bias_ih_l3_reverse = lstm_layer.bias_ih_l3_reverse

    quantized_lstm_layer.weight_hh_l0 = lstm_layer.weight_hh_l0
    quantized_lstm_layer.weight_hh_l0_reverse = lstm_layer.weight_hh_l0_reverse
    quantized_lstm_layer.weight_hh_l1 = lstm_layer.weight_hh_l1
    quantized_lstm_layer.weight_hh_l1_reverse = lstm_layer.weight_hh_l1_reverse
    quantized_lstm_layer.weight_hh_l2 = lstm_layer.weight_hh_l2
    quantized_lstm_layer.weight_hh_l2_reverse = lstm_layer.weight_hh_l2_reverse
    quantized_lstm_layer.weight_hh_l3 = lstm_layer.weight_hh_l3
    quantized_lstm_layer.weight_hh_l3_reverse = lstm_layer.weight_hh_l3_reverse

    quantized_lstm_layer.weight_ih_l0 = lstm_layer.weight_ih_l0
    quantized_lstm_layer.weight_ih_l0_reverse = lstm_layer.weight_ih_l0_reverse
    quantized_lstm_layer.weight_ih_l1 = lstm_layer.weight_ih_l1
    quantized_lstm_layer.weight_ih_l1_reverse = lstm_layer.weight_ih_l1_reverse
    quantized_lstm_layer.weight_ih_l2 = lstm_layer.weight_ih_l2
    quantized_lstm_layer.weight_ih_l2_reverse = lstm_layer.weight_ih_l2_reverse
    quantized_lstm_layer.weight_ih_l3 = lstm_layer.weight_ih_l3
    quantized_lstm_layer.weight_ih_l3_reverse = lstm_layer.weight_ih_l3_reverse

    return quantized_lstm_layer


def QuantEmbedding(embedding_layer, quant_desc_weight):

    quantized_embedding_layer = QuantEmb(
        num_embeddings=embedding_layer.num_embeddings,
        embedding_dim=embedding_layer.embedding_dim,
        padding_idx=embedding_layer.padding_idx,
        quant_desc_weight=quant_desc_weight
    )

    quantized_embedding_layer.weight = embedding_layer.weight

    return quantized_embedding_layer



def QuantGRUCell(gru_cell, quant_desc_input, quant_desc_weight):
    quantized_gru_cell = quant_nn.QuantGRUCell(
        input_size=gru_cell.input_size,
        hidden_size=gru_cell.hidden_size,
        bias=gru_cell.bias,
        quant_desc_input=quant_desc_input,
        quant_desc_weight=quant_desc_weight
    )

    # Copy weights and biases
    quantized_gru_cell.weight_ih = gru_cell.weight_ih
    quantized_gru_cell.weight_hh = gru_cell.weight_hh
    if gru_cell.bias:
        quantized_gru_cell.bias_ih = gru_cell.bias_ih
        quantized_gru_cell.bias_hh = gru_cell.bias_hh

    return quantized_gru_cell

def QuantConv1d(conv1d_layer, quant_desc_input, quant_desc_weight):
    quantized_conv1d_layer = quant_nn.QuantConv1d(
        in_channels=conv1d_layer.in_channels,
        out_channels=conv1d_layer.out_channels,
        kernel_size=conv1d_layer.kernel_size,
        stride=conv1d_layer.stride,
        padding=conv1d_layer.padding,
        dilation=conv1d_layer.dilation,
        groups=conv1d_layer.groups,
        bias=conv1d_layer.bias is not None,
        padding_mode=conv1d_layer.padding_mode,
        quant_desc_input=quant_desc_input,
        quant_desc_weight=quant_desc_weight
    )

    # Copy weights and biases from the original conv1d_layer
    quantized_conv1d_layer.weight = conv1d_layer.weight
    if conv1d_layer.bias is not None:
        quantized_conv1d_layer.bias = conv1d_layer.bias

    return quantized_conv1d_layer

def QuantLayerNorm(layer_norm_layer, quant_desc_input):
    quantized_layer_norm = quant_nn.QuantLayerNorm(
        normalized_shape=layer_norm_layer.normalized_shape,
        eps=layer_norm_layer.eps,
        elementwise_affine=layer_norm_layer.elementwise_affine,
        quant_desc_input=quant_desc_input
    )

    if layer_norm_layer.elementwise_affine:
        quantized_layer_norm.weight = layer_norm_layer.weight
        quantized_layer_norm.bias = layer_norm_layer.bias

    return quantized_layer_norm

def QuantBatchNorm1d(batchnorm1d_layer, quant_desc_input):
    quantized_batchnorm1d_layer = quant_nn.QuantBatchNorm1d(
        num_features=batchnorm1d_layer.num_features,
        eps=batchnorm1d_layer.eps,
        momentum=batchnorm1d_layer.momentum,
        affine=batchnorm1d_layer.affine,
        track_running_stats=batchnorm1d_layer.track_running_stats,
        quant_desc_input=quant_desc_input
    )

    # Copy weights and biases from the original batchnorm1d_layer
    if batchnorm1d_layer.affine:
        quantized_batchnorm1d_layer.weight = batchnorm1d_layer.weight
        quantized_batchnorm1d_layer.bias = batchnorm1d_layer.bias

    if batchnorm1d_layer.track_running_stats:
        quantized_batchnorm1d_layer.running_mean = batchnorm1d_layer.running_mean
        quantized_batchnorm1d_layer.running_var = batchnorm1d_layer.running_var

    return quantized_batchnorm1d_layer

def quantize_layer(layer):
    #input_quant_descriptor = QuantDescriptor(num_bits=8, calib_method='histogram')
    #conv_weight_quant_descriptor = QuantDescriptor(num_bits=8)
    #weight_quant_descriptor = QuantDescriptor(num_bits=8, calib_method='histogram')
    if isinstance(layer, torch.nn.Conv2d):
        return QuantConv(layer, quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                         quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL)
    elif isinstance(layer, torch.nn.Linear):
        return QuantLinear(layer, quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                           quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
    elif isinstance(layer, torch.nn.MaxPool2d):
        return QuantMaxPool2d(layer, quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
    elif isinstance(layer, torch.nn.LSTM):
        return QuantLSTM(layer, quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                         quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
    elif isinstance(layer, torch.nn.Embedding):
        return QuantEmbedding(layer, quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
    elif isinstance(layer, torch.nn.GRUCell):
        return QuantGRUCell(layer, quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                            quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
    elif isinstance(layer, torch.nn.Conv1d):
        return QuantConv1d(layer, quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                           quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_CONV1D_WEIGHT_PER_CHANNEL)
    elif isinstance(layer, torch.nn.LayerNorm):
        return QuantLayerNorm(layer, quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
    elif isinstance(layer, torch.nn.BatchNorm1d):
        return QuantBatchNorm1d(layer, quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
    else:
        return None
    """
    input_quant_descriptor = QuantDescriptor(num_bits=8, calib_method='histogram')
    conv_weight_quant_descriptor = QuantDescriptor(num_bits=8, axis=(0))
    weight_quant_descriptor = QuantDescriptor(num_bits=8, calib_method='histogram')

    if isinstance(layer, torch.nn.Conv2d):
        return QuantConv(layer, quant_desc_input=input_quant_descriptor,
                         quant_desc_weight=conv_weight_quant_descriptor)
    elif isinstance(layer, torch.nn.Linear):
        return QuantLinear(layer, quant_desc_input=input_quant_descriptor,
                           quant_desc_weight=weight_quant_descriptor)
    elif isinstance(layer, torch.nn.MaxPool2d):
        return QuantMaxPool2d(layer, quant_desc_input=input_quant_descriptor)
    elif isinstance(layer, torch.nn.LSTM):
        return QuantLSTM(layer, quant_desc_input=input_quant_descriptor,
                         quant_desc_weight=weight_quant_descriptor)
    elif isinstance(layer, torch.nn.Embedding):
        return QuantEmbedding(layer, quant_desc_weight=weight_quant_descriptor)
    elif isinstance(layer, torch.nn.GRUCell):
        return QuantGRUCell(layer, quant_desc_input=input_quant_descriptor,
                            quant_desc_weight=weight_quant_descriptor)
    elif isinstance(layer, torch.nn.Conv1d):
        return QuantConv1d(layer, quant_desc_input=input_quant_descriptor,
                           quant_desc_weight=conv_weight_quant_descriptor)
    elif isinstance(layer, torch.nn.LayerNorm):
        return QuantLayerNorm(layer, quant_desc_input=input_quant_descriptor)
    elif isinstance(layer, torch.nn.BatchNorm1d):
        return QuantBatchNorm1d(layer, quant_desc_input=input_quant_descriptor)
    else:
        return None
   """

def collect_stats(quantized_layers, data_loader, num_batches, model):
    # Enable calibrators
    for layer in quantized_layers:
        if hasattr(layer, 'input_quantizer') and layer.input_quantizer._calibrator is not None:
            layer.input_quantizer.disable_quant()
            layer.input_quantizer.enable_calib()
        if hasattr(layer, 'weight_quantizer') and layer.weight_quantizer._calibrator is not None:
            layer.weight_quantizer.disable_quant()
            layer.weight_quantizer.enable_calib()

    # Feed data through the network
    for i, batch in tqdm(enumerate(data_loader), total=num_batches):
        print(batch)
        #for layer in quantized_layers:
            #loss = model.compute_forward(batch, sb.Stage.TEST)
        loss = model.compute_forward(batch, sb.Stage.TEST)
        if i >= num_batches:
            break
        continue

    # Disable calibrators
    for layer in quantized_layers:
        if hasattr(layer, 'input_quantizer') and layer.input_quantizer._calibrator is not None:
            layer.input_quantizer.enable_quant()
            layer.input_quantizer.disable_calib()
        if hasattr(layer, 'weight_quantizer') and layer.weight_quantizer._calibrator is not None:
            layer.weight_quantizer.enable_quant()
            layer.weight_quantizer.disable_calib()


def compute_amax(quantized_layers, **kwargs):
    # Load calibration results for each quantized layer
    for layer in quantized_layers:
        if hasattr(layer, 'input_quantizer') and layer.input_quantizer._calibrator is not None:
            if isinstance(layer.input_quantizer._calibrator, calib.MaxCalibrator):
                layer.input_quantizer.load_calib_amax()
            else:
                layer.input_quantizer.load_calib_amax(**kwargs)

        if hasattr(layer, 'weight_quantizer') and layer.weight_quantizer._calibrator is not None:
            if isinstance(layer.weight_quantizer._calibrator, calib.MaxCalibrator):
                layer.weight_quantizer.load_calib_amax()
            else:
                layer.weight_quantizer.load_calib_amax(**kwargs)
                     


def quantize(asr, train_data):



    quantized_layers = []

    asr.modules['enc']['CNN']['block_0']['conv_1'].conv = quantize_layer(
        asr.modules['enc']['CNN']['block_0']['conv_1'].conv)
    quantized_layers.append(asr.modules['enc']['CNN']['block_0']['conv_1'].conv)
    asr.modules['enc']['CNN']['block_0']['conv_2'].conv = quantize_layer(
        asr.modules['enc']['CNN']['block_0']['conv_2'].conv)
    quantized_layers.append(asr.modules['enc']['CNN']['block_0']['conv_2'].conv)
    
    asr.modules['enc']['CNN']['block_1']['conv_1'].conv = quantize_layer(
        asr.modules['enc']['CNN']['block_1']['conv_1'].conv)
    quantized_layers.append(asr.modules['enc']['CNN']['block_1']['conv_1'].conv)
    asr.modules['enc']['CNN']['block_1']['conv_2'].conv = quantize_layer(
        asr.modules['enc']['CNN']['block_1']['conv_2'].conv)
    quantized_layers.append(asr.modules['enc']['CNN']['block_1']['conv_2'].conv)

    asr.modules['enc']['CNN']['block_0']['pooling'].pool_layer = quantize_layer(
        asr.modules['enc']['CNN']['block_0']['pooling'].pool_layer)
    quantized_layers.append(asr.modules['enc']['CNN']['block_0']['pooling'].pool_layer)
    asr.modules['enc']['CNN']['block_1']['pooling'].pool_layer = quantize_layer(
        asr.modules['enc']['CNN']['block_1']['pooling'].pool_layer)
    quantized_layers.append(asr.modules['enc']['CNN']['block_1']['pooling'].pool_layer)
    
    
    asr.modules['enc']['CNN']['block_0']['norm_1'].norm = quantize_layer(asr.modules['enc']['CNN']['block_0']['norm_1'].norm)
    quantized_layers.append(asr.modules['enc']['CNN']['block_0']['norm_1'].norm)
    asr.modules['enc']['CNN']['block_0']['norm_2'].norm = quantize_layer(asr.modules['enc']['CNN']['block_0']['norm_2'].norm)
    quantized_layers.append(asr.modules['enc']['CNN']['block_0']['norm_2'].norm)

    asr.modules['enc']['CNN']['block_1']['norm_1'].norm = quantize_layer(asr.modules['enc']['CNN']['block_1']['norm_1'].norm)
    quantized_layers.append(asr.modules['enc']['CNN']['block_1']['norm_1'].norm)
    asr.modules['enc']['CNN']['block_1']['norm_2'].norm = quantize_layer(asr.modules['enc']['CNN']['block_1']['norm_2'].norm)
    quantized_layers.append(asr.modules['enc']['CNN']['block_1']['norm_2'].norm)

    asr.modules['enc']['DNN']['block_0']['linear'].w = quantize_layer(asr.modules['enc']['DNN']['block_0']['linear'].w)
    quantized_layers.append(asr.modules['enc']['DNN']['block_0']['linear'].w)
    asr.modules['enc']['DNN']['block_1']['linear'].w = quantize_layer(asr.modules['enc']['DNN']['block_1']['linear'].w)
    quantized_layers.append(asr.modules['enc']['DNN']['block_1']['linear'].w)

    
    asr.modules['enc']['DNN']['block_0']['norm'].norm = quantize_layer(asr.modules['enc']['DNN']['block_0']['norm'].norm)
    quantized_layers.append(asr.modules['enc']['DNN']['block_0']['norm'].norm)

    asr.modules['enc']['DNN']['block_1']['norm'].norm = quantize_layer(asr.modules['enc']['DNN']['block_1']['norm'].norm)
    quantized_layers.append(asr.modules['enc']['DNN']['block_1']['norm'].norm)

    #sr.modules['enc']['RNN'].rnn = quantize_layer(asr.modules['enc']['RNN'].rnn)
    #uantized_layers.append(asr.modules['enc']['RNN'].rnn)

    asr_brain.modules['enc'].time_pooling.pool_layer = quantize_layer(asr_brain.modules['enc'].time_pooling.pool_layer)
    quantized_layers.append(asr_brain.modules['enc'].time_pooling.pool_layer)
    
    #asr.modules['emb'].Embedding = quantize_layer(asr.modules['emb'].Embedding)
    #quantized_layers.append(asr.modules['emb'].Embedding)
    asr.modules['dec'].rnn.rnn_cells[0] = quantize_layer(asr.modules['dec'].rnn.rnn_cells[0])
    quantized_layers.append(asr.modules['dec'].rnn.rnn_cells[0])
    asr.modules['dec'].proj = quantize_layer(asr.modules['dec'].proj)
    quantized_layers.append(asr.modules['dec'].proj)

    asr.modules['dec'].attn.conv_loc = quantize_layer(asr.modules['dec'].attn.conv_loc)
    quantized_layers.append(asr.modules['dec'].attn.conv_loc)
    asr.modules['dec'].attn.mlp_attn = quantize_layer( asr.modules['dec'].attn.mlp_attn)
    quantized_layers.append(asr.modules['dec'].attn.mlp_attn)
    asr.modules['dec'].attn.mlp_dec = quantize_layer(asr.modules['dec'].attn.mlp_dec)
    quantized_layers.append(asr.modules['dec'].attn.mlp_dec)
    asr.modules['dec'].attn.mlp_enc = quantize_layer(asr.modules['dec'].attn.mlp_enc)
    quantized_layers.append(asr.modules['dec'].attn.mlp_enc)
    asr.modules['dec'].attn.mlp_loc = quantize_layer(asr.modules['dec'].attn.mlp_loc)
    quantized_layers.append(asr.modules['dec'].attn.mlp_loc)
    asr.modules['dec'].attn.mlp_out = quantize_layer(asr.modules['dec'].attn.mlp_out)
    quantized_layers.append(asr.modules['dec'].attn.mlp_out)
    
    #asr.modules['ctc_lin'].w = quantize_layer(asr.modules['ctc_lin'].w)
    #quantized_layers.append(asr.modules['ctc_lin'].w)
    asr.modules['seq_lin'].w = quantize_layer(asr.modules['seq_lin'].w)
    quantized_layers.append(asr.modules['seq_lin'].w)


    with torch.no_grad():
        collect_stats(quantized_layers, train_data, num_batches=8, model=asr)
        compute_amax(quantized_layers, method="mse")

    asr.modules.to(asr.device)

    # The next line ensures that both tensors marked as parameters and standard tensors,
    # such as those used in InputNormalization, are placed on the right device.
    for module in asr.modules:
        if hasattr(asr.modules[module], "to"):
            asr.modules[module] = asr.modules[module].to(asr.device)
        
        
if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {"batch_sampler": train_bsampler}
    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )
    
    import os

    # Testing
    k = 'test-clean'
    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])
    asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
    
    quantized_layers = []
    asr_brain.modules['enc']['RNN'].rnn = quantize_layer(asr_brain.modules['enc']['RNN'].rnn)
    quantized_layers.append(asr_brain.modules['enc']['RNN'].rnn)
    
    quant_dataloader = asr_brain.make_dataloader(
            train_data, stage=sb.Stage.TRAIN, **train_dataloader_opts
        )

  
    with torch.no_grad():
        collect_stats(quantized_layers, quant_dataloader, num_batches=8, model=asr_brain)
        compute_amax(quantized_layers, method="mse")
    
    asr_brain.quant_evaluate(
            test_datasets[k], train_data,
            test_loader_kwargs=hparams["test_dataloader_opts"],
            train_loader_kwargs=hparams["train_dataloader_opts"],
            min_key="WER",
        )
    

