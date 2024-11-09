"""Classes for Nnet3 alignment"""
from __future__ import annotations

import logging
import pathlib
import typing
from functools import partial

from _kalpy.fstext import VectorFst
from _kalpy.matrix import FloatMatrix
from _kalpy.nnet3 import NnetSimpleComputationOptions, nnet3_align_compiled
from kalpy.gmm.align import GmmAligner
from kalpy.gmm.data import Alignment
from kalpy.nnet3.utils import read_nnet3_model

logger = logging.getLogger("kalpy.align")
logger.setLevel(logging.DEBUG)
logger.write = lambda msg: logger.info(msg) if msg != "\n" else None
logger.flush = lambda: None


class Nnet3Aligner(GmmAligner):
    def __init__(
        self,
        acoustic_model_path: typing.Union[pathlib.Path, str],
        acoustic_scale: float = 1.0,
        transition_scale: float = 1.0,
        self_loop_scale: float = 1.0,
        **kwargs,
    ):
        super(Nnet3Aligner, self).__init__(
            acoustic_model_path,
            read_model_fn=read_nnet3_model,
            acoustic_scale=acoustic_scale,
            transition_scale=transition_scale,
            self_loop_scale=self_loop_scale,
            **kwargs,
        )

        decodable_opts = NnetSimpleComputationOptions()
        decodable_opts.acoustic_scale = acoustic_scale

        acoustic_model_path = pathlib.Path(acoustic_model_path).parent
        decodable_opts.extra_left_context = int(
            open(f"{acoustic_model_path}/left_context").read().strip()
        )
        decodable_opts.extra_right_context = int(
            open(f"{acoustic_model_path}/right_context").read().strip()
        )
        decodable_opts.frames_per_chunk = 100
        self.decodable_opts = decodable_opts

    def align_utterance(
        self, training_graph: VectorFst, features: FloatMatrix, utterance_id: str = None
    ) -> typing.Optional[Alignment]:
        (
            alignment,
            words,
            likelihood,
            per_frame_log_likelihoods,
            successful,
            retried,
        ) = nnet3_align_compiled(
            self.transition_model,
            self.acoustic_model,
            self.decodable_opts,
            training_graph,
            features,
            acoustic_scale=self.acoustic_scale,
            transition_scale=self.transition_scale,
            self_loop_scale=self.self_loop_scale,
            beam=self.beam,
            retry_beam=self.retry_beam,
            careful=self.careful,
        )
        if not successful:
            return None
        if retried and utterance_id:
            logger.debug(f"Retried {utterance_id}")
        return Alignment(utterance_id, alignment, words, likelihood, per_frame_log_likelihoods)
