from __future__ import annotations

import json
import logging
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_AUDIO_DIR = PROJECT_ROOT / "generated_audio"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
SAMPLE_RATE = 24_000


class _InvalidCharacterWarningFilter(logging.Filter):
    """Hide ChatTTS normalization notices that are not synthesis failures."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("found invalid characters:")


def set_seed(seed: int = 42) -> None:
    """Set common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_directories() -> None:
    for directory in (GENERATED_AUDIO_DIR, TABLES_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch

        report.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
            }
        )
    except Exception as exc:
        report["torch_error"] = str(exc)

    for package_name in ("ChatTTS", "torchaudio", "librosa", "soundfile"):
        try:
            module = __import__(package_name)
            report[package_name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            report[f"{package_name}_error"] = str(exc)

    return report


def read_text_manifest(path: str | Path = PROJECT_ROOT / "data" / "text_manifest.csv") -> pd.DataFrame:
    manifest = pd.read_csv(path)
    manifest["char_count"] = manifest["text"].str.len()
    manifest["word_count"] = manifest["text"].str.split().str.len()
    manifest["has_numbers"] = manifest["text"].str.contains(r"\d", regex=True)
    manifest["has_special_tokens"] = manifest["text"].str.contains(r"\[.*?\]", regex=True)
    return manifest


def load_chattts(compile_model: bool = False) -> Any:
    try:
        import ChatTTS
        from transformers.utils import logging as transformers_logging
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ChatTTS nije dostupan u Python okruzenju koje koristi notebook. "
            f"Trenutni interpreter je: {sys.executable}. "
            "U terminalu iz foldera 2-audio-tts pokreni: "
            "source .venv/bin/activate && python -m ipykernel install --user "
            "--name chattts-tts --display-name 'Python (chattts-tts)', "
            "pa u notebook-u izaberi kernel Python (chattts-tts)."
        ) from exc

    # Widget progress bars are not reliably preserved by every notebook
    # frontend. Model loading still works normally with this display disabled.
    transformers_logging.disable_progress_bar()
    chattts_logger = logging.getLogger("ChatTTS.core")
    if not any(
        isinstance(log_filter, _InvalidCharacterWarningFilter)
        for log_filter in chattts_logger.filters
    ):
        chattts_logger.addFilter(_InvalidCharacterWarningFilter())

    chat = ChatTTS.Chat()
    loaded = chat.load(source="local", custom_path=PROJECT_ROOT, compile=compile_model)
    if not loaded:
        raise RuntimeError(
            "ChatTTS model nije uspesno ucitan. Proveri da li folder "
            f"{PROJECT_ROOT / 'asset'} sadrzi sve model fajlove."
        )
    return chat


def normalize_audio_array(audio: Any) -> np.ndarray:
    audio_array = np.asarray(audio, dtype=np.float32)
    audio_array = np.squeeze(audio_array)
    if audio_array.ndim != 1:
        audio_array = audio_array.reshape(-1)
    peak = float(np.max(np.abs(audio_array))) if audio_array.size else 0.0
    if peak > 1.0:
        audio_array = audio_array / peak
    return audio_array


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


@dataclass
class SynthesisResult:
    text_id: str
    language: str
    category: str
    style: str
    text: str
    output_path: str
    synthesis_time_s: float
    audio_duration_s: float
    real_time_factor: float
    sample_rate: int


def synthesize_one(
    chat: Any,
    text_id: str,
    text: str,
    language: str,
    category: str,
    style: str,
    params_refine_text: Any | None = None,
    params_infer_code: Any | None = None,
    skip_refine_text: bool = False,
    show_progress: bool = False,
) -> SynthesisResult:
    start = time.perf_counter()
    infer_kwargs: dict[str, Any] = {"skip_refine_text": skip_refine_text}
    if not skip_refine_text:
        infer_kwargs["params_refine_text"] = (
            params_refine_text
            if params_refine_text is not None
            else chat.RefineTextParams(show_tqdm=show_progress)
        )
    infer_kwargs["params_infer_code"] = (
        params_infer_code
        if params_infer_code is not None
        else chat.InferCodeParams(show_tqdm=show_progress)
    )

    wavs = chat.infer([text], **infer_kwargs)
    synthesis_time_s = time.perf_counter() - start

    audio = normalize_audio_array(wavs[0])
    output_path = GENERATED_AUDIO_DIR / f"{text_id}.wav"
    save_audio(output_path, audio, SAMPLE_RATE)

    audio_duration_s = float(len(audio) / SAMPLE_RATE) if len(audio) else 0.0
    real_time_factor = synthesis_time_s / audio_duration_s if audio_duration_s > 0 else np.nan
    return SynthesisResult(
        text_id=text_id,
        language=language,
        category=category,
        style=style,
        text=text,
        output_path=str(output_path),
        synthesis_time_s=float(synthesis_time_s),
        audio_duration_s=audio_duration_s,
        real_time_factor=float(real_time_factor),
        sample_rate=SAMPLE_RATE,
    )


def synthesize_manifest(
    chat: Any,
    manifest: pd.DataFrame,
    show_progress: bool = False,
) -> list[SynthesisResult]:
    results: list[SynthesisResult] = []
    for row in manifest.itertuples(index=False):
        results.append(
            synthesize_one(
                chat=chat,
                text_id=row.text_id,
                text=row.text,
                language=row.language,
                category=row.category,
                style=row.style,
                show_progress=show_progress,
            )
        )
    return results


def signal_features(audio_path: str | Path) -> dict[str, float]:
    import librosa

    audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    if audio.size == 0:
        return {
            "rms_db": np.nan,
            "zcr_mean": np.nan,
            "spectral_centroid_mean": np.nan,
            "pitch_mean_hz": np.nan,
        }

    rms = librosa.feature.rms(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    try:
        pitch, _, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )
        pitch_mean = float(np.nanmean(pitch)) if np.any(~np.isnan(pitch)) else np.nan
    except Exception:
        pitch_mean = np.nan

    rms_mean = float(np.mean(rms)) if rms.size else np.nan
    rms_db = float(20 * np.log10(max(rms_mean, 1e-12)))
    return {
        "rms_db": rms_db,
        "zcr_mean": float(np.mean(zcr)) if zcr.size else np.nan,
        "spectral_centroid_mean": float(np.mean(centroid)) if centroid.size else np.nan,
        "pitch_mean_hz": pitch_mean,
    }


def build_results_table(results: list[SynthesisResult], manifest: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame([result.__dict__ for result in results])
    if base.empty:
        return base

    features = []
    for row in base.itertuples(index=False):
        features.append({"text_id": row.text_id, **signal_features(row.output_path)})
    feature_table = pd.DataFrame(features)

    text_meta = manifest[
        ["text_id", "char_count", "word_count", "has_numbers", "has_special_tokens"]
    ].copy()
    return base.merge(text_meta, on="text_id", how="left").merge(
        feature_table, on="text_id", how="left"
    )


def read_subjective_scores(
    path: str | Path = PROJECT_ROOT / "data" / "subjective_scores_template.csv",
) -> pd.DataFrame:
    return pd.read_csv(path)


def merge_subjective_scores(results_table: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if results_table.empty:
        return results_table
    return results_table.merge(scores, on="text_id", how="left")


def save_table(table: pd.DataFrame, filename: str) -> Path:
    path = TABLES_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_waveform_and_mel(audio_path: str | Path, figure_path: str | Path) -> None:
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    librosa.display.waveshow(audio, sr=sample_rate, ax=axes[0])
    axes[0].set_title("Waveform")
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
    )
    axes[1].set_title("Mel spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    fig.tight_layout()
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
