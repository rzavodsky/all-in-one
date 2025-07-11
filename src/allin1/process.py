from demucs.api import Separator
import torch
import numpy as np
from typing import Callable

from allin1.postprocessing.functional import postprocess_functional_structure
from allin1.postprocessing.metrical import postprocess_metrical_structure
from allin1.postprocessing.tempo import estimate_tempo_from_beats
from .typings import AnalysisResult
from .models import load_pretrained_model
from madmom.audio.signal import FramedSignalProcessor, SignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor


def _separator_callback(data):
    progress_callback: Callable[[float], None] = data["progress_callback"]
    if data['state'] != 'end': return
    pos: float = data['segment_offset'] / data['audio_length']
    progress_callback(pos * 0.5)


def _extract_spectrograms(data: dict[str, np.ndarray], sr: int):
    signal = SignalProcessor(sample_rate=sr)
    frames = FramedSignalProcessor(
        frame_size=2048,
        fps=int(44100 / 441)
    )
    stft = ShortTimeFourierTransformProcessor()  # caching FFT window
    filt = FilteredSpectrogramProcessor(
        num_bands=12,
        fmin=30,
        fmax=17000,
        norm_filters=True
    )
    spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
    processor = SequentialProcessor([signal, frames, stft, filt, spec])

    result = [
        processor(data["bass"]),
        processor(data["drums"]),
        processor(data["other"]),
        processor(data["vocals"]),
    ]
    return np.stack(result)


def analyze_array(
        audio_data: np.ndarray,
        sr: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        include_embeddings: bool = False,
        progress_callback: Callable[[float], None] | None = None
) -> AnalysisResult:
    if not progress_callback:
        progress_callback = lambda _: None

    sep = Separator(callback=_separator_callback, callback_arg={
        "progress_callback": progress_callback
    })
    _, demixed = sep.separate_tensor(torch.from_numpy(audio_data), sr)
    demixed_np: dict[str, np.ndarray] = {}

    for stem in demixed:
        data = demixed[stem]
        # Convert to mono
        data = data.mean(dim=0)
        # Prevent clipping
        data /= max(1.01 * data.abs().max().item(), 1)
        # Convert to numpy int16 array
        data = data.numpy()
        data *= 2**15 - 1
        data = data.astype(np.int16)
        demixed_np[stem] = data

    spec = _extract_spectrograms(demixed_np, sep.samplerate)
    spec = torch.from_numpy(spec).unsqueeze(0).to(device)
    progress_callback(0.5)

    model = load_pretrained_model(
        model_name="harmonix-all",
        device=device,
    )

    with torch.no_grad():
        logits = model(spec, progress_callback=lambda x: progress_callback(x*0.49 + 0.5))
        progress_callback(0.99)
        metrical_structure = postprocess_metrical_structure(logits, model.cfg)
        functional_structure = postprocess_functional_structure(logits, model.cfg)
        bpm = estimate_tempo_from_beats(metrical_structure['beats'])
    progress_callback(1)
    assert bpm is not None
    result = AnalysisResult(
        path=None,
        bpm=bpm,
        segments=functional_structure,
        **metrical_structure,
    )
    if include_embeddings:
        result.embeddings = logits.embeddings[0].cpu().numpy()
    return result
