import pathlib
import typing

from _kalpy.hmm import TransitionModel
from _kalpy.nnet3 import AmNnetSimple
from _kalpy.util import Input, Output


def read_nnet3_model(
    model_path: typing.Union[str, pathlib.Path]
) -> typing.Tuple[TransitionModel, AmNnetSimple]:
    ki = Input()
    ki.Open(str(model_path), True)
    transition_model = TransitionModel()
    transition_model.Read(ki.Stream(), True)
    acoustic_model = AmNnetSimple()
    acoustic_model.Read(ki.Stream(), True)
    ki.Close()
    return transition_model, acoustic_model


def write_nnet3_model(
    model_path: typing.Union[str, pathlib.Path],
    transition_model: TransitionModel,
    acoustic_model: AmNnetSimple,
    binary: bool = True,
) -> None:
    ko = Output(str(model_path), binary)
    transition_model.Write(ko.Stream(), binary)
    acoustic_model.Write(ko.Stream(), binary)
    ko.Close()
