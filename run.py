#!/usr/bin/env python3
"""Mithi v2 Flash: text to speech in 11 Indian languages.

Command line (inside this folder):
    python run.py --lang hi --voice charvi --text "à¤†à¤œ à¤®à¥Œà¤¸à¤® à¤•à¤¿à¤¤à¤¨à¤¾ à¤…à¤šà¥à¤›à¤¾ à¤¹à¥ˆ!" --out hello.wav
    python run.py --lang en --voice satvik --text "Hello! How are you today?" --out hello_en.wav
    python run.py --device cpu --threads 4 --lang hi --text "à¤¨à¤®à¤¸à¥à¤¤à¥‡! à¤†à¤ª à¤•à¥ˆà¤¸à¥‡ à¤¹à¥ˆà¤‚?" --out hello_cpu.wav

Python (inside this folder):
    from run import Mithi
    tts = Mithi()                                     # loads the model from this folder
    tts.save(tts.synthesize("à¤¨à¤®à¤¸à¥à¤¤à¥‡! à¤†à¤ª à¤•à¥ˆà¤¸à¥‡ à¤¹à¥ˆà¤‚?", lang="hi", voice="charvi"), "namaste.wav")

Languages: hi en bn gu kn ml mr or pa ta te
Voices:    satvik anirban nakul vasant siddhant (male), charvi ahana vaidehi malini jahnavi (female)

Includes code from Kokoro (github.com/hexgrad/kokoro, Apache-2.0) and the Maya text normaliser (Apache-2.0).
"""
import argparse
import importlib.abc
import importlib.util
import linecache
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- embedded package (loaded from memory)
_SOURCES = {
    'mithi_tts': (True, r'''"""Mithi: Indian-language text-to-speech (11 languages, 10 voices)."""
from .tts import SR, Mithi  # noqa: F401

__version__ = "1.0.0"
'''),
    'mithi_tts.frontend': (False, r'''"""Text front end: Maya text normaliser -> espeak-ng / misaki G2P -> Kokoro phoneme string, plus the words indic-bert
reads and the word index of every phoneme token.

    norm, ps = phonemes("à¤†à¤œ à¤®à¥Œà¤¸à¤® à¤•à¤¿à¤¤à¤¨à¤¾ à¤…à¤šà¥à¤›à¤¾ à¤¹à¥ˆ!", "hi")
"""
import json
import os
import re
import unicodedata

from .text_norm import normalize_text

LANGS = {"hi": "Hindi", "en": "English", "bn": "Bengali", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
         "mr": "Marathi", "or": "Odia", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu"}

# Kokoro-82M phoneme vocabulary (char -> id) plus the breath token the writer inserts (id 7, never emitted by G2P)
from ._here import HERE as _HERE
VOCAB = json.load(open(os.path.join(_HERE, "config.json"), encoding="utf-8"))["kokoro"]["vocab"]
BREATH, BREATH_ID = "â‚", 7
VOCAB = {**VOCAB, BREATH: BREATH_ID}

NUKTA = "à¤¼"
PUNCT = ',.!?;:â€”â€¦"â€œâ€()'
SPLIT_PUNCT = re.compile(r"([,.!?;:â€”â€¦\"â€œâ€()])")
LATIN_RUN = re.compile(r"([A-Za-z][A-Za-z']*(?:\s+[A-Za-z][A-Za-z']*)*)")
# espeak-ng phones some Indic languages use that Kokoro's vocabulary lacks -> the closest Kokoro phone
IPA_ALIASES = {"É­": "l", "Ê‰": "u", "Ì©": "", "+": ""}
PAUSE_IDS = {VOCAB[c] for c in " ,.!?;:â€”â€¦" if c in VOCAB}
SPACE = VOCAB[" "]
PUNCT_IDS = {VOCAB[c] for c in PUNCT if c in VOCAB}

# ---------------------------------------------------------------- G2P
_g2p = {}


def _short_espeak_path():
    """espeak-ng cannot open a data folder whose path is longer than about 160 characters (deep virtualenvs). In that
    case copy the data (19 MB, once) to a short folder and point espeak there."""
    try:
        import shutil
        import tempfile
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        data = espeakng_loader.get_data_path()
        if len(data) < 140:
            return
        for base in (os.path.expanduser("~/.cache/mithi"), os.path.join(tempfile.gettempdir(), "mithi")):
            short = os.path.join(base, "espeak-ng-data")
            if len(short) >= 140:
                continue
            try:
                if not os.path.exists(os.path.join(short, "phontab")):
                    os.makedirs(base, exist_ok=True)
                    tmp = short + f".tmp{os.getpid()}"
                    shutil.copytree(data, tmp)
                    os.replace(tmp, short)
            except OSError:
                continue
            EspeakWrapper.set_data_path(short)
            return
    except Exception:
        pass


def _init_g2p(lang):
    from misaki import en, espeak
    _short_espeak_path()
    if lang not in LANGS:
        raise ValueError(f"unsupported language: {lang} (one of {', '.join(LANGS)})")
    if lang != "en":
        _g2p[lang] = espeak.EspeakG2P(language=lang)
    _g2p["en"] = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False))


def restore_nukta(text, text_norm):
    """The Maya normaliser drops the nukta (à¤ªà¥‡à¤¡à¤¼ -> à¤ªà¥‡à¤¡); put it back wherever the original word had one."""
    nfd = lambda w: unicodedata.normalize("NFD", w)  # noqa: E731
    table = {}
    for w in re.findall(r"[à¤€-à¥¿]+", text):
        if NUKTA in nfd(w):
            table[unicodedata.normalize("NFC", nfd(w).replace(NUKTA, ""))] = unicodedata.normalize("NFC", w)
    if not table:
        return text_norm
    return re.sub(r"[à¤€-à¥¿]+", lambda m: table.get(m.group(), m.group()), text_norm)


# Legacy-font typing leaves two vowel signs in a row (à¤›à¤¾à¥‡à¤Ÿà¤¾, à¤•à¤¾à¥…à¤•à¥à¤¸); espeak-ng crashes on some of them.
MATRA_FIX = [("à¤¾à¥…", "à¥‰"), ("à¥…à¤¾", "à¥‰"), ("à¤¾à¥‡", "à¥‹"), ("à¥‡à¤¾", "à¥‹"), ("à¤¾à¥ˆ", "à¥Œ"), ("à¥ˆà¤¾", "à¥Œ")]
DOUBLE_SIGN = re.compile(r"([à¤¾-à¥Œ])[à¤¾-à¥Œ]+")


def fix_matras(t):
    for bad, good in MATRA_FIX:
        t = t.replace(bad, good)
    return DOUBLE_SIGN.sub(r"\1", t)


def _phonemize_hi(chunk):
    chunk = re.sub(r"['\-â€“]+", " ", chunk).strip()        # espeak-ng hi crashes on a run starting with "'-"
    if not chunk:
        return ""
    ps = _g2p["hi"](chunk)[0]
    # espeak-ng hi writes à¤¡à¤¼/à¤¢à¤¼ as "r."; inside a punctuation-free chunk any "." is that marker
    return ps.replace("r.", "É½").replace(".", "")


def _phonemize(text, text_norm, lang):
    t = restore_nukta(text, text_norm) if lang in ("hi", "mr") else text_norm
    if lang in ("hi", "mr"):
        t = fix_matras(t)
    t = t.replace("à¥¤", ".").replace("à¥¥", ".")
    t = re.sub(r"(?<=\w)-(?=\w)", " ", t)
    out = ""
    for piece in SPLIT_PUNCT.split(t):
        if not piece:
            continue
        if piece in PUNCT:
            out = out.rstrip() + piece + " "
            continue
        words = []
        for run in LATIN_RUN.split(piece):
            run = run.strip()
            if not run:
                continue
            if LATIN_RUN.fullmatch(run):
                words.append(_g2p["en"](run)[0])
            elif lang == "hi":
                words.append(_phonemize_hi(run))
            else:
                words.append(_g2p[lang](run)[0])
        if words:
            out += " ".join(w for w in words if w) + " "
    out = re.sub(r"\s+", " ", out).strip()
    for old, new in IPA_ALIASES.items():
        out = out.replace(old, new)
    return "".join(c for c in out if c in VOCAB)


def phonemes(text, lang):
    """text -> (normalised text, Kokoro phoneme string)."""
    if lang not in _g2p:
        _init_g2p(lang)
    norm = normalize_text(text, lang)
    return norm, _phonemize(text, norm, lang)


# ---------------------------------------------------------------- words for indic-bert
def words_of(text, text_norm):
    """The words the meaning encoder reads (same transformations as _phonemize), punctuation attached to the word
    before it. The n-th space-separated phoneme group is the n-th word."""
    t = fix_matras(restore_nukta(text, text_norm)).replace("à¥¤", ".").replace("à¥¥", ".")
    t = re.sub(r"(?<=\w)-(?=\w)", " ", t)
    t = re.sub(r"['\-â€“]+", " ", t)
    out = ""
    for piece in SPLIT_PUNCT.split(t):
        if not piece:
            continue
        out = out.rstrip() + piece + " " if piece in PUNCT else out + piece.strip() + " "
    words = []
    for g in (w for w in re.sub(r"\s+", " ", out).split(" ") if w):
        if all(c in PUNCT for c in g):          # a punctuation-only group joins the word before it
            if words:
                words[-1] += g
        else:
            words.append(g)
    return words


def token_words(ids):
    """Word index per token (BOS/EOS and spaces -1, punctuation / breath -> the word before) and the word count."""
    out, w, in_word, last = [], -1, False, -1
    for k, t in enumerate(ids):
        if k == 0 or k == len(ids) - 1 or t == SPACE:
            out.append(-1)
            in_word = False
        elif t == BREATH_ID or t in PUNCT_IDS:
            out.append(last)
        else:
            if not in_word:
                w += 1
                in_word = True
            out.append(w)
            last = w
    return out, w + 1


# ---------------------------------------------------------------- long text
def split_text(text, lang, limit=400):
    """The whole text if its phonemes fit, else sentences (and, if still too long, comma pieces) under the limit.
    Odia à­± is written à¬¬ (espeak-ng spells à­± words letter by letter)."""
    text = " ".join(text.split())
    if lang == "or":
        text = text.replace("à­±", "à¬¬")
    if len(phonemes(text, lang)[1]) <= limit:
        return [text]
    parts = [s.strip() for s in re.split(r"(?<=[.!?à¥¤à¥¥])\s+", text) if s.strip()]
    out = []
    for s in parts:
        if len(phonemes(s, lang)[1]) <= limit:
            out.append(s)
            continue
        cur = ""
        for piece in re.split(r"(?<=[,;:])\s+", s):
            if cur and len(phonemes(cur + " " + piece, lang)[1]) > limit:
                out.append(cur)
                cur = piece
            else:
                cur = (cur + " " + piece).strip()
        if cur:
            out.append(cur)
    return out
'''),
    'mithi_tts.indic_encoder': (False, r'''"""Meaning encoder for the prosody writer: ai4bharat/indic-bert (ALBERT) with its word table trimmed to the 105,229
pieces our languages use (21.3 M parameters, outputs identical to the full model).

Preprocessing matches indic-bert's pretraining (ALBERT's preprocess_text): collapse spaces, NFKD, drop combining marks
(in Indic scripts: virama and nukta; vowel signs stay), lowercase. SentencePiece is called directly (the transformers
AlbertTokenizer also strips the vowel signs) and its ids are remapped to the trimmed table.
"""
import unicodedata

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from transformers import AlbertModel

CLS, SEP = 2, 3


def preprocess(text):
    text = unicodedata.normalize("NFKD", " ".join(text.split()))
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


class IndicTextEncoder(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.sp = spm.SentencePieceProcessor(model_file=f"{model_dir}/spiece.model")
        self.register_buffer("old2new", torch.from_numpy(np.load(f"{model_dir}/old2new.npy")), persistent=False)
        self.bert = AlbertModel.from_pretrained(model_dir, dtype=torch.float32)
        self.dim = self.bert.config.hidden_size

    def pieces(self, words):
        ids, owner = [CLS], [-1]
        for w, word in enumerate(words):
            p = self.sp.encode(preprocess(word))
            ids += p
            owner += [w] * len(p)
        ids.append(SEP)
        owner.append(-1)
        return self.old2new[torch.tensor(ids)], torch.tensor(owner)

    def word_vectors(self, text):
        """One vector per space-separated word of `text` (mean of its pieces' last hidden states)."""
        words = text.split()
        ids, owner = self.pieces(words)
        h = self.bert(input_ids=ids[None].to(self.old2new.device)).last_hidden_state[0]
        out = torch.zeros(len(words), self.dim, device=h.device, dtype=h.dtype)
        cnt = torch.zeros(len(words), 1, device=h.device, dtype=h.dtype)
        sel = owner >= 0
        out.index_add_(0, owner[sel].to(h.device), h[sel.to(h.device)])
        cnt.index_add_(0, owner[sel].to(h.device), torch.ones(int(sel.sum()), 1, device=h.device, dtype=h.dtype))
        return out / cnt.clamp(min=1), words
'''),
    'mithi_tts.kokoro': (True, r'''"""Kokoro-82M text encoder and iSTFTNet decoder (hexgrad/kokoro, Apache-2.0), vendored unchanged except for imports."""
'''),
    'mithi_tts.kokoro.custom_stft': (False, r'''import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomSTFT(nn.Module):
    """
    STFT/iSTFT without unfold/complex ops, using conv1d and conv_transpose1d.

    - forward STFT => Real-part conv1d + Imag-part conv1d
    - inverse STFT => Real-part conv_transpose1d + Imag-part conv_transpose1d + sum
    - avoids F.unfold, so easier to export to ONNX
    - uses replicate or constant padding for 'center=True' to approximate 'reflect' 
      (reflect is not supported for dynamic shapes in ONNX)
    """

    def __init__(
        self,
        filter_length=800,
        hop_length=200,
        win_length=800,
        window="hann",
        center=True,
        pad_mode="replicate",  # or 'constant'
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = filter_length
        self.center = center
        self.pad_mode = pad_mode

        # Number of frequency bins for real-valued STFT with onesided=True
        self.freq_bins = self.n_fft // 2 + 1

        # Build window
        assert window == 'hann', window
        window_tensor = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
        if self.win_length < self.n_fft:
            # Zero-pad up to n_fft
            extra = self.n_fft - self.win_length
            window_tensor = F.pad(window_tensor, (0, extra))
        elif self.win_length > self.n_fft:
            window_tensor = window_tensor[: self.n_fft]
        self.register_buffer("window", window_tensor)

        # Precompute forward DFT (real, imag)
        # PyTorch stft uses e^{-j 2 pi k n / N} => real=cos(...), imag=-sin(...)
        n = np.arange(self.n_fft)
        k = np.arange(self.freq_bins)
        angle = 2 * np.pi * np.outer(k, n) / self.n_fft  # shape (freq_bins, n_fft)
        dft_real = np.cos(angle)
        dft_imag = -np.sin(angle)  # note negative sign

        # Combine window and dft => shape (freq_bins, filter_length)
        # We'll make 2 conv weight tensors of shape (freq_bins, 1, filter_length).
        forward_window = window_tensor.numpy()  # shape (n_fft,)
        forward_real = dft_real * forward_window  # (freq_bins, n_fft)
        forward_imag = dft_imag * forward_window

        # Convert to PyTorch
        forward_real_torch = torch.from_numpy(forward_real).float()
        forward_imag_torch = torch.from_numpy(forward_imag).float()

        # Register as Conv1d weight => (out_channels, in_channels, kernel_size)
        # out_channels = freq_bins, in_channels=1, kernel_size=n_fft
        self.register_buffer(
            "weight_forward_real", forward_real_torch.unsqueeze(1)
        )
        self.register_buffer(
            "weight_forward_imag", forward_imag_torch.unsqueeze(1)
        )

        # Precompute inverse DFT
        # Real iFFT formula => scale = 1/n_fft, doubling for bins 1..freq_bins-2 if n_fft even, etc.
        # For simplicity, we won't do the "DC/nyquist not doubled" approach here. 
        # If you want perfect real iSTFT, you can add that logic. 
        # This version just yields good approximate reconstruction with Hann + typical overlap.
        inv_scale = 1.0 / self.n_fft
        n = np.arange(self.n_fft)
        angle_t = 2 * np.pi * np.outer(n, k) / self.n_fft  # shape (n_fft, freq_bins)
        idft_cos = np.cos(angle_t).T  # => (freq_bins, n_fft)
        idft_sin = np.sin(angle_t).T  # => (freq_bins, n_fft)

        # Multiply by window again for typical overlap-add
        # We also incorporate the scale factor 1/n_fft
        inv_window = window_tensor.numpy() * inv_scale
        backward_real = idft_cos * inv_window  # (freq_bins, n_fft)
        backward_imag = idft_sin * inv_window

        # We'll implement iSTFT as real+imag conv_transpose with stride=hop.
        self.register_buffer(
            "weight_backward_real", torch.from_numpy(backward_real).float().unsqueeze(1)
        )
        self.register_buffer(
            "weight_backward_imag", torch.from_numpy(backward_imag).float().unsqueeze(1)
        )
        


    def transform(self, waveform: torch.Tensor):
        """
        Forward STFT => returns magnitude, phase
        Output shape => (batch, freq_bins, frames)
        """
        # waveform shape => (B, T).  conv1d expects (B, 1, T).
        # Optional center pad
        if self.center:
            pad_len = self.n_fft // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)

        x = waveform.unsqueeze(1)  # => (B, 1, T)
        # Convolution to get real part => shape (B, freq_bins, frames)
        real_out = F.conv1d(
            x,
            self.weight_forward_real,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        # Imag part
        imag_out = F.conv1d(
            x,
            self.weight_forward_imag,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )

        # magnitude, phase
        magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-14)
        phase = torch.atan2(imag_out, real_out)
        # Handle the case where imag_out is 0 and real_out is negative to correct ONNX atan2 to match PyTorch
        # In this case, PyTorch returns pi, ONNX returns -pi
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase[correction_mask] = torch.pi
        return magnitude, phase


    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor, length=None):
        """
        Inverse STFT => returns waveform shape (B, T).
        """
        # magnitude, phase => (B, freq_bins, frames)
        # Re-create real/imag => shape (B, freq_bins, frames)
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)

        # conv_transpose wants shape (B, freq_bins, frames). We'll treat "frames" as time dimension
        # so we do (B, freq_bins, frames) => (B, freq_bins, frames)
        # But PyTorch conv_transpose1d expects (B, in_channels, input_length)
        real_part = real_part  # (B, freq_bins, frames)
        imag_part = imag_part

        # real iSTFT => convolve with "backward_real", "backward_imag", and sum
        # We'll do 2 conv_transpose calls, each giving (B, 1, time),
        # then add them => (B, 1, time).
        real_rec = F.conv_transpose1d(
            real_part,
            self.weight_backward_real,  # shape (freq_bins, 1, filter_length)
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        imag_rec = F.conv_transpose1d(
            imag_part,
            self.weight_backward_imag,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        # sum => (B, 1, time)
        waveform = real_rec - imag_rec  # typical real iFFT has minus for imaginary part

        # If we used "center=True" in forward, we should remove pad
        if self.center:
            pad_len = self.n_fft // 2
            # Because of transposed convolution, total length might have extra samples
            # We remove `pad_len` from start & end if possible
            waveform = waveform[..., pad_len:-pad_len]

        # If a specific length is desired, clamp
        if length is not None:
            waveform = waveform[..., :length]

        # shape => (B, T)
        return waveform

    def forward(self, x: torch.Tensor):
        """
        Full STFT -> iSTFT pass: returns time-domain reconstruction.
        Same interface as your original code.
        """
        mag, phase = self.transform(x)
        return self.inverse(mag, phase, length=x.shape[-1])
'''),
    'mithi_tts.kokoro.istftnet': (False, r'''# ADAPTED from https://github.com/yl4579/StyleTTS2/blob/main/Modules/istftnet.py
from .custom_stft import CustomSTFT
from torch.nn.utils.parametrizations import weight_norm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/yl4579/StyleTTS2/blob/main/Modules/utils.py
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        # affine should be False, however there's a bug in the old torch.onnx.export (not newer dynamo) that causes the channel dimension to be lost if affine=False. When affine is true, there's additional learnably parameters. This shouldn't really matter setting it to True, since we're in inference mode
        self.norm = nn.InstanceNorm1d(num_features, affine=True)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaINResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super(AdaINResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                  padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                                  padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                                  padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        self.adain1 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        self.adain2 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        return x


class TorchSTFT(nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        assert window == 'hann', window
        self.window = torch.hann_window(win_length, periodic=True, dtype=torch.float32)

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device),
            return_complex=True)
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))
        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class SineGen(nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(torch.pi) or cos(0)
    """
    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * torch.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1
        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1/self.upsample_scale, mode="linear").transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
            phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
            sines = torch.sin(phase)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation
            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            # get the sines
            sines = torch.cos(i_phase * 2 * torch.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        # fundamental component
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp
        # generate uv signal
        # uv = torch.ones(f0.shape)
        # uv = uv * (f0 > self.voiced_threshold)
        uv = self._f02uv(f0)
        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        #        for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """
    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)
        # to merge source harmonics into a single excitation
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class Generator(nn.Module):
    def __init__(self, style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, disable_complex=False):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
                    sampling_rate=24000,
                    upsample_scale=math.prod(upsample_rates) * gen_istft_hop_size,
                    harmonic_num=8, voiced_threshod=10)
        self.f0_upsamp = nn.Upsample(scale_factor=math.prod(upsample_rates) * gen_istft_hop_size)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                   k, u, padding=(k-u)//2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes,resblock_dilation_sizes)):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(nn.Conv1d(
                    gen_istft_n_fft + 2, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
                self.noise_res.append(AdaINResBlock1(c_cur, 7, [1,3,5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(c_cur, 11, [1,3,5], style_dim))
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(nn.Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = (
            CustomSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)
            if disable_complex
            else TorchSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)
        )

    def forward(self, x, s, f0):
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
            har_source, noi_source, uv = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1) 
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        return self.stft.inverse(spec, phase)


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) * torch.rsqrt(torch.tensor(2))
        return out


class Decoder(nn.Module):
    def __init__(self, dim_in, style_dim, dim_out, 
                 resblock_kernel_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 resblock_dilation_sizes,
                 upsample_kernel_sizes,
                 gen_istft_n_fft, gen_istft_hop_size,
                 disable_complex=False):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList()
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True))
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(512, 64, kernel_size=1)))
        self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates, 
                                   upsample_initial_channel, resblock_dilation_sizes, 
                                   upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, disable_complex=disable_complex)

    def forward(self, asr, F0_curve, N, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        x = self.generator(x, s, F0_curve)
        return x
'''),
    'mithi_tts.kokoro.modules': (False, r'''# https://github.com/yl4579/StyleTTS2/blob/main/models.py
from .istftnet import AdainResBlk1d
from torch.nn.utils.parametrizations import weight_norm
from transformers import AlbertModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)  # [B, T, chn]
        lengths = input_lengths if input_lengths.device == torch.device('cpu') else input_lengths.to('cpu')
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad
        x.masked_fill_(m, 0.0)
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid,nlayers=nlayers, dropout=dropout)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        m = m.unsqueeze(1)
        lengths = text_lengths if text_lengths.device == torch.device('cpu') else text_lengths.to('cpu')
        x = nn.utils.rnn.pack_padded_sequence(d, lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
        x_pad[:, :x.shape[1], :] = x
        x = x_pad
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))
        en = (d.transpose(-1, -2) @ alignment)
        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, d_model // 2, num_layers=1, batch_first=True, bidirectional=True))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                lengths = text_lengths if text_lengths.device == torch.device('cpu') else text_lengths.to('cpu')
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad

        return x.transpose(-1, -2)


# https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state
'''),
    'mithi_tts.prosody': (False, r'''"""Prosody writer: reads a sentence (phoneme tokens + indic-bert word meaning + speaker + language) and writes Maya-style
prosody one phoneme token at a time, as discrete codes; decode() turns the codes into the per-token durations and the
frame-level F0 / energy the Kokoro decoder takes (2 frames per 25 ms unit).

Codes per token (P5): dur, uv0, uv1, p1..p5 (pitch, 0.5-semitone bins re 100 Hz), e1..e3 (energy, 0.25 bins),
then breath_next (1 = the writer inserts a breath token after this one).
The writer predicts voicing explicitly and pitch / energy as changes from the previous point (v2 design), and its depth
heads predict the codes of one token in order, each seeing the ones before it.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .frontend import BREATH_ID

# ---------------------------------------------------------------- code tables
DUR_MAX, UV_MAX = 63, 31
ST_LO, ST_STEP, N_ST = -10.0, 0.5, 81            # -10 .. +30 semitones
PITCH_NONE = N_ST                                # extra class: no voiced frame
E_LO, E_STEP, N_E = -6.0, 0.25, 65               # -6 .. +10
K_PITCH, J_ENERGY = 5, 3                         # P5: 5 pitch + 3 energy points
P5_COLS = 3 + K_PITCH + J_ENERGY                 # 11 = index of breath_next in what sample() returns


def _interp(points, n):
    """n frame values through k points placed at the centres of k equal parts."""
    k = len(points)
    if n == 0:
        return np.zeros(0)
    pos = (np.arange(k) + 0.5) / k * n - 0.5
    return np.interp(np.arange(n), pos, points)


def decode(codes):
    """P5 codes [T, 11] -> dur [T] units, f0 [2*sum(dur)] Hz, energy [2*sum(dur)]."""
    k = K_PITCH
    dur = codes[:, 0].astype(np.int64)
    n = 2 * int(dur.sum())
    f0, en, s = np.zeros(n, np.float32), np.zeros(n, np.float32), 0
    for t, d in enumerate(dur):
        if d == 0:
            continue
        a, b = 2 * s, 2 * (s + d)
        s += d
        en[a:b] = _interp(E_LO + E_STEP * codes[t, 3 + k:].astype(np.float64), b - a)
        p = codes[t, 3:3 + k]
        if (p == PITCH_NONE).any():
            continue
        v0, v1 = a + min(int(codes[t, 1]), b - a), b - min(int(codes[t, 2]), b - a)
        if v1 > v0:
            st = _interp(ST_LO + ST_STEP * p.astype(np.float64), v1 - v0)
            f0[v0:v1] = 100 * 2 ** (st / 12)
    return dur, f0, en


# ---------------------------------------------------------------- writer
NAMES = ["dur", "voiced", "uv0", "uv1", "d1", "d2", "d3", "d4", "d5", "f1", "f2", "f3", "breath"]
WRITER_LANGS = ["hi", "en", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "as"]
D_P, D_E = 80, 64
REF_P = int(round((0 - ST_LO) / ST_STEP))           # 0 st re 100 Hz: the reference before the first voiced token
REF_E = int(round((0 - E_LO) / E_STEP))
ABS_CLASSES = [DUR_MAX + 1, 2, UV_MAX + 1, UV_MAX + 1] + [N_ST + 1] * 5 + [N_E] * 3 + [2]
REL_CLASSES = [DUR_MAX + 1, 2, UV_MAX + 1, UV_MAX + 1] + [2 * D_P + 1] * 5 + [2 * D_E + 1] * 3 + [2]
VOICED, BREATH_NEXT = 1, 12                         # columns of the 13-column absolute layout


def abs_to_p5(codes_abs):
    """absolute layout [N, 13] -> (P5 codes [N, 11], breath_next [N])."""
    return torch.cat([codes_abs[:, :1], codes_abs[:, 2:12]], 1), codes_abs[:, BREATH_NEXT]


class ProsodyWriter(nn.Module):
    def __init__(self, n_tok=178, word_dim=768, d=384, heads=6, enc_layers=4, dec_layers=4, n_spk=2,
                 n_lang=len(WRITER_LANGS), depth_hid=256, dropout=0.1):
        super().__init__()
        self.d = d
        self.tok = nn.Embedding(n_tok, d)
        self.word = nn.Linear(word_dim, d)
        self.no_word = nn.Parameter(torch.zeros(d))
        self.spk = nn.Embedding(n_spk, d)
        self.lang = nn.Embedding(n_lang, d)
        self.enc_pos = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True),
                                         enc_layers, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d)
        self.code_emb = nn.ModuleList([nn.Embedding(c, d) for c in ABS_CLASSES])
        self.start = nn.Parameter(torch.zeros(d))
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True),
                                         dec_layers)
        self.dec_norm = nn.LayerNorm(d)
        self.depth = nn.ModuleList([nn.Sequential(nn.Linear(d, depth_hid), nn.GELU(), nn.Linear(depth_hid, c)) for c in REL_CLASSES])

    def read(self, enc_ids, word_vec, has_word, spk, lang, enc_pad):
        x = (self.tok(enc_ids) + torch.where(has_word[..., None], self.word(word_vec), self.no_word)
             + (self.spk(spk) + self.lang(lang))[:, None])
        x = x.masked_fill(enc_pad[..., None], 0)
        x = x + self.enc_pos(x.transpose(1, 2)).transpose(1, 2)
        return self.enc_norm(self.enc(x, src_key_padding_mask=enc_pad))

    def _prev_codes(self, ctx):
        e = sum(emb(ctx[..., k]) for k, emb in enumerate(self.code_emb))
        return torch.cat([self.start.expand(ctx.shape[0], 1, self.d), e[:, :-1]], 1)

    def write(self, H, enc_pad, dec_ids, dec_map, ctx, dec_pad):
        x = torch.gather(H, 1, dec_map[..., None].expand(-1, -1, self.d)) + self.tok(dec_ids) + self._prev_codes(ctx)
        causal = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), 1)
        h = self.dec(x, H, tgt_mask=causal, tgt_key_padding_mask=dec_pad, memory_key_padding_mask=enc_pad, tgt_is_causal=True)
        return self.dec_norm(h)

    @staticmethod
    def _abs_from_rel(k, c, abs_row, prev_p5, prev_e3):
        if k == 4:
            return (prev_p5 + c - D_P).clamp(0, N_ST - 1)
        if 5 <= k <= 8:
            return (abs_row[k - 1] + c - D_P).clamp(0, N_ST - 1)
        if k == 9:
            return (prev_e3 + c - D_E).clamp(0, N_E - 1)
        if 10 <= k <= 11:
            return (abs_row[k - 1] + c - D_E).clamp(0, N_E - 1)
        return c

    @torch.no_grad()
    def sample(self, H, enc_ids, pause_ids, temp=0.8, top_p=0.9, max_breaths=None, head_temp=None):
        """Write one sentence (batch of 1), token by token. head_temp = {code name: temperature} (0 = most likely).
        Returns (dec_ids [Nd] incl. inserted breaths, P5 codes + breath_next [Nd, 12])."""
        dev = H.device
        dec_ids, dec_map, rows = [], [], []
        enc_pad = torch.zeros(1, H.shape[1], dtype=torch.bool, device=dev)
        queue = [(int(t), i) for i, t in enumerate(enc_ids.tolist())]
        n_breath, prev_p5, prev_e3 = 0, REF_P, REF_E
        while queue:
            tok, i = queue.pop(0)
            dec_ids.append(tok)
            dec_map.append(i)
            ctx = torch.tensor(rows + [[0] * len(NAMES)], device=dev)[None]
            ids = torch.tensor(dec_ids, device=dev)[None]
            h = self.write(H, enc_pad, ids, torch.tensor(dec_map, device=dev)[None], ctx,
                           torch.zeros_like(ids, dtype=torch.bool))[0, -1]
            row, acc = [], h
            for k, head in enumerate(self.depth):
                t = (head_temp or {}).get(NAMES[k], temp)
                if 2 <= k <= 8 and row[VOICED] == 0:
                    a = PITCH_NONE if k >= 4 else 0
                else:
                    c = _sample(head(acc), t, 1.0 if k == BREATH_NEXT else top_p)
                    a = int(self._abs_from_rel(k, torch.tensor(c), torch.tensor(row + [0]), torch.tensor(prev_p5), torch.tensor(prev_e3))) \
                        if 4 <= k <= 11 else c
                if k == BREATH_NEXT and (tok not in pause_ids or tok == BREATH_ID or not queue
                                         or (max_breaths is not None and n_breath >= max_breaths)):
                    a = 0
                row.append(a)
                acc = acc + self.code_emb[k](torch.tensor(a, device=dev))
            rows.append(row)
            if row[VOICED] == 1:
                prev_p5 = row[8]
            prev_e3 = row[11]
            if row[BREATH_NEXT] == 1:
                queue.insert(0, (BREATH_ID, i))
                n_breath += 1
        out = torch.tensor(rows)
        p5, breath = abs_to_p5(out)
        return torch.tensor(dec_ids), torch.cat([p5, breath[:, None]], 1)


def _sample(logits, temp, top_p):
    if temp <= 0:
        return int(logits.argmax())
    p = F.softmax(logits / temp, -1)
    sp, idx = p.sort(descending=True)
    keep = sp.cumsum(-1) - sp < top_p
    sp = sp * keep
    return int(idx[torch.multinomial(sp / sp.sum(), 1)])
'''),
    'mithi_tts.text_norm': (True, r'''"""Text normalisation for Maya3 (numbers, currency, dates, OTPs, URLs, ... -> spoken words, per language).

    from mithi_tts.text_norm import normalise_text
    normalise_text("Your OTP is 482913 [laughs]", "en")   # 'Your OTP is four, eight, two, nine, one, three [laughs]'

Same behaviour as the normalise_text wrapper the release evals used:
  * inline [tags] (e.g. [laughs], [sighs]) are protected: the text between them is normalised piece by piece;
  * ALL-CAPS Latin words that look like names (>= 5 letters with >= 2 vowels, or 4 letters with >= 2 vowels) are
    title-cased first, so the normaliser and the model read them as words; short / vowel-poor tokens (OTP, NEFT,
    HDFC, KYC) stay acronyms;
  * each piece goes through the bundled normaliser mithi_tts.text_norm.tn (pure Python, standard library only);
  * if the normaliser raises on a piece, that piece is kept unchanged (the eval wrapper's second fallback, a
    separate benchmark pipeline, is not part of the release);
  * whitespace is collapsed and the result stripped;
  * `rules=True` (default) adds the targeted pre/post rules of mithi_tts.text_norm.rules (two-digit-year dates, card expiry,
    FY ranges, spaced card/phone digits, PIN codes, version strings, rupees+paise in every language, the comma
    before an inline [tag]). `rules=False` is the exact behaviour of the release evals before these rules were added.
`lang` is a Maya3 language code (as bn en gu hi kn ml mr or pa ta te); for an unknown code the normaliser raises,
so the text comes back with only the caps fold and whitespace collapse applied.
"""
from __future__ import annotations

import logging
import re

from mithi_tts.text_norm import lexicon as _lexicon
from mithi_tts.text_norm import rules as _rules

__all__ = ["normalise_text", "normalize_text", "SOURCE"]

SOURCE = "Maya Research text normaliser, upstream commit fe5e6ec"

_TAG = re.compile(r"(\[[a-z ]+\])")
_CAPS = re.compile(r"\b[A-Z]{4,}\b")
_VOWELS = set("AEIOU")
_TN = None

log = logging.getLogger("mithi_tts.text_norm")


def _caps_fold(t):
    def f(m):
        w = m.group(0)
        if len(w) >= 5 and sum(ch in _VOWELS for ch in w) >= 2 or (len(w) == 4 and sum(ch in _VOWELS for ch in w) >= 2):
            return w.title()
        return w
    return _CAPS.sub(f, t)


def _load_tn():
    global _TN
    if _TN is None:
        from mithi_tts.text_norm.tn.normalize import normalize
        _TN = normalize
    return _TN


def normalise_text(text: str, lang: str, rules: bool = True) -> str:
    tn = _load_tn()
    _rules.install(rules)
    _lexicon.install(rules)
    out = []
    segs = _TAG.split(text)
    for i, seg in enumerate(segs):
        if not seg or _TAG.fullmatch(seg):
            out.append(seg)
            continue
        s = _caps_fold(seg)
        try:
            s2 = tn(_rules.pre(s, lang) if rules else s, lang)
            if rules:
                s2 = _rules.post(s2, seg, lang, before_tag=any(_TAG.fullmatch(x) for x in segs[i + 1:i + 2]))
        except Exception as exc:   # never fail a request on normalisation: keep this piece as written
            log.debug("normaliser fallback (%s: %s)", type(exc).__name__, exc)
            s2 = s
        lead = seg[:len(seg) - len(seg.lstrip())]
        trail = seg[len(seg.rstrip()):]
        out.append(lead + s2.strip() + trail)
    return re.sub(r"\s+", " ", "".join(out)).strip()


normalize_text = normalise_text
'''),
    'mithi_tts.text_norm.lexicon': (False, r'''"""Language-table corrections applied around the vendored normaliser (mithi_tts.text_norm.tn), found by a multi-language
normaliser benchmark.

The vendored copy stays byte-identical. `install(True)` (called by mithi_tts.text_norm.normalise_text with rules on) patches
the module-level language tables / one Normalizer method; `install(False)` restores the originals exactly, so
`normalise_text(..., rules=False)` keeps the eval-time behaviour.

  L1 as_lexicon  Assamese inherited Bengali grammar tables: Bengali à¤° (U+09B0) instead of Assamese à§° (U+09F0) in the
                 number words (à¦¤à§à¦°à¦¿à¦¶ -> à¦¤à§à§°à¦¿à¦¶), Bengali ordinals / symbol words / letter names / clock words
                 (à¦šà¦¤à§à¦°à§à¦¥, à¦…à§à¦¯à¦¾à¦Ÿ à¦¦à§à¦¯ à¦°à§‡à¦Ÿ, à¦†à¦°, à¦¸à¦•à¦¾à¦², à¦¦à§à¦ªà§à¦°, à¦°à¦¾à¦¤, "{h}à¦Ÿà¦¾", "à¦à¦° à¦¬à¦°à§à¦—" ...). Replaced with Assamese forms;
                 Assamese clock markers (à¦¬à¦œà¦¾à¦¤ / à¦¬à¦œà¦¾à§° / à¦¬à¦œà¦¾à¦²à§ˆ) and period words (à¦ªà§à§±à¦¾, à¦†à¦¬à§‡à¦²à¦¿, à¦¬à¦¿à¦¯à¦¼à¦²à¦¿, à§°à¦¾à¦¤à¦¿ ...)
                 are recognised, and postpositions glue to the number noun (à¦ªà¦žà§à¦šà¦¾à¦¶ à¦Ÿà¦•à¦¾ à§° -> à¦ªà¦žà§à¦šà¦¾à¦¶ à¦Ÿà¦•à¦¾à§°).
  L2 per_unit    "km/h", "MB/s": the per word was the English "per" in bn as gu kn ml mr pa ta, and the
                 denominator was a spelled letter ("à¦•à§‡ à¦à¦® per à¦à¦‡à¦š"); now the language's per word + hour / second.
  L3 uri_hosts   bare domains with sub-domains (e-receipt.hdfcbank.com, netbanking.nationalbank.in) were only
                 matched from the last label, so "e-receipt.hdfcbank" stayed raw; a bare domain that ends a sentence
                 ("irctc.co.in.") was not matched at all; the local part of a UPI ID (rahul.deshmukh@okaxis) stayed
                 raw. All three are now spelled like other URLs.
"""
from __future__ import annotations

import re
import types

__all__ = ["install", "AS_ONES"]

_R, _RR = "à¦°", "à§°"      # Bengali ra, Assamese ra


def _as_script(s):
    return s.replace(_R, _RR) if isinstance(s, str) else s


# Assamese 0-99: AI4Bharat indic-numtowords data (tn/assamese_numbers.py) with the Bengali ra fixed to à§°
# (à¦Šà¦¨à¦¤à§à¦°à¦¿à¦¶..à¦†à¦ à¦¤à§à¦°à¦¿à¦¶ -> à¦Šà¦¨à¦¤à§à§°à¦¿à¦¶..à¦†à¦ à¦¤à§à§°à¦¿à¦¶) and two malformed entries corrected: 45 à¦ªà¦à¦žà§à¦šà¦²à§à¦²à¦¿à¦¶ -> à¦ªà¦žà§à¦šà¦²à§à¦²à¦¿à¦¶,
# 71 à¦à¦¸à¦¤à§à¦¤à§° -> à¦à¦•à¦¸à¦¤à§à¦¤à§°.
def _as_ones():
    from mithi_tts.text_norm.tn.assamese_numbers import ONES
    ones = [_as_script(w) for w in ONES]
    ones[45] = "à¦ªà¦žà§à¦šà¦²à§à¦²à¦¿à¦¶"
    ones[71] = "à¦à¦•à¦¸à¦¤à§à¦¤à§°"
    return ones


AS_ONES = None   # filled on first install

_AS_SYM = {
    "at": "à¦à¦Ÿ à¦¦à§à¦¯ à§°à§‡à¦Ÿ", "underscore": "à¦†à¦£à§à¦¡à¦¾à§°à¦¸à§à¦•'à§°", "slash": "à¦¶à§à¦²à§‡à¦š", "hash": "à¦¹à§‡à¦š", "amp": "à¦à¦£à§à¦¡",
    "squared": "à¦¬à§°à§à¦—", "sqrt": "à¦¬à§°à§à¦—à¦®à§‚à¦²", "lt": "à¦¤à¦•à§ˆ à¦•à¦®", "gt": "à¦¤à¦•à§ˆ à¦¬à§‡à¦›à¦¿",
    "half": "à¦†à¦§à¦¾", "quarter": "à¦à¦• à¦šà¦¤à§à§°à§à¦¥à¦¾à¦‚à¦¶", "three_quarters": "à¦¤à¦¿à¦¨à¦¿ à¦šà¦¤à§à§°à§à¦¥à¦¾à¦‚à¦¶",
    "degree": "à¦¡à¦¿à¦—à§à§°à§€", "celsius": "à¦¡à¦¿à¦—à§à§°à§€ à¦šà§‡à¦²à¦›à¦¿à¦¯à¦¼à¦¾à¦›", "fahrenheit": "à¦¡à¦¿à¦—à§à§°à§€ à¦«à¦¾à§°à§‡à¦¨à¦¹à¦¾à¦‡à¦Ÿ",
    "second": "à¦›à§‡à¦•à§‡à¦£à§à¦¡", "cent": "à¦šà§‡à¦£à§à¦Ÿ", "per": "à¦ªà§à§°à¦¤à¦¿",
}
_AS_PERIOD_DETECT = {
    "à¦ªà§à§±à¦¾": "morning", "à¦ªà§à§±à¦¾à§°": "morning", "à¦ªà§à§±à¦¾à¦¤à§‡": "morning", "à§°à¦¾à¦¤à¦¿à¦ªà§à§±à¦¾": "morning",
    "à¦¦à§à¦ªà§°à§€à¦¯à¦¼à¦¾": "afternoon", "à¦¦à§à¦ªà§°à§€à¦¯à¦¼à¦¾à§°": "afternoon", "à¦¦à§à¦ªà§°à§€à¦¯à¦¼à¦¾à¦¤": "afternoon",
    "à¦†à¦¬à§‡à¦²à¦¿": "evening", "à¦†à¦¬à§‡à¦²à¦¿à§°": "evening", "à¦†à¦¬à§‡à¦²à¦¿à¦¤à§‡": "evening", "à¦¬à¦¿à¦¯à¦¼à¦²à¦¿": "evening", "à¦¬à¦¿à¦¯à¦¼à¦²à¦¿à§°": "evening",
    "à¦¸à¦¨à§à¦§à¦¿à¦¯à¦¼à¦¾": "evening", "à¦¸à¦¨à§à¦§à¦¿à¦¯à¦¼à¦¾à§°": "evening", "à¦¸à¦¨à§à¦§à¦¿à¦¯à¦¼à¦¾à¦¤à§‡": "evening",
    "à§°à¦¾à¦¤à¦¿": "night", "à§°à¦¾à¦¤à¦¿à§°": "night", "à§°à¦¾à¦¤à¦¿à¦¤à§‡": "night",
}
_AS_ATTACH = {"à¦Ÿà¦•à¦¾", "à¦ªà¦‡à¦šà¦¾", "à¦¡à¦²à¦¾à§°", "à¦¶à¦¤à¦¾à¦‚à¦¶", "à¦®à¦¿à¦¨à¦¿à¦Ÿ", "à¦˜à¦£à§à¦Ÿà¦¾", "à¦›à§‡à¦•à§‡à¦£à§à¦¡", "à¦¬à¦œà¦¾", "à¦¡à¦¿à¦—à§à§°à§€", "à¦•à¦¿à¦²à§‹à¦®à¦¿à¦Ÿà¦¾à§°"}
_AS_PP = ("à§°", "à¦¤", "à¦²à§ˆ", "à¦¤à¦•à§ˆ", "à§°à§‡", "à¦¤à§‡", "à¦²à§ˆà¦•à§‡", "à§°à¦ªà§°à¦¾")


def _as_attach(self, rep, tok):
    """Assamese case endings glue to the noun without sandhi: à¦Ÿà¦•à¦¾+à§° -> à¦Ÿà¦•à¦¾à§°, à¦®à¦¿à¦¨à¦¿à¦Ÿ+à¦¤ -> à¦®à¦¿à¦¨à¦¿à¦Ÿà¦¤."""
    t = tok.lstrip("-")
    last = rep.rsplit(" ", 1)[-1]
    if last not in _AS_ATTACH or t not in _AS_PP:
        return None
    return rep + t


_PER = {"bn": "à¦ªà§à¦°à¦¤à¦¿", "gu": "àªªà«àª°àª¤àª¿", "mr": "à¤ªà¥à¤°à¤¤à¤¿", "pa": "à¨ªà©à¨°à¨¤à©€",
        "ta": "à®ªà¯†à®°à¯", "kn": "à²ªà²°à³", "ml": "à´ªàµ†àµ¼"}            # hi or te already have one; as: _AS_SYM
_PER_UNITS = {   # denominators of X/h, X/s (H, S are not units on their own, so only "km/h"-style reads change)
    "en": {"H": "hour", "S": "second"},
    "hi": {"H": "à¤˜à¤‚à¤Ÿà¤¾", "S": "à¤¸à¥‡à¤•à¤‚à¤¡"}, "mr": {"H": "à¤¤à¤¾à¤¸", "S": "à¤¸à¥‡à¤•à¤‚à¤¦"},
    "bn": {"H": "à¦˜à¦£à§à¦Ÿà¦¾", "S": "à¦¸à§‡à¦•à§‡à¦¨à§à¦¡"}, "as": {"H": "à¦˜à¦£à§à¦Ÿà¦¾", "S": "à¦›à§‡à¦•à§‡à¦£à§à¦¡"},
    "gu": {"H": "àª•àª²àª¾àª•", "S": "àª¸à«‡àª•àª¨à«àª¡"}, "pa": {"H": "à¨˜à©°à¨Ÿà¨¾", "S": "à¨¸à¨•à¨¿à©°à¨Ÿ"}, "or": {"H": "à¬˜à¬£à­à¬Ÿà¬¾", "S": "à¬¸à­‡à¬•à­‡à¬£à­à¬¡"},
    "ta": {"H": "à®¹à®µà®°à¯", "S": "à®šà¯†à®•à®£à¯à®Ÿà¯"}, "te": {"S": "à°¸à±†à°•à°‚à°¡à±"},          # te already has H
    "kn": {"H": "à²…à²µà²°à³", "S": "à²¸à³†à²•à³†à²‚à²¡à³"}, "ml": {"H": "à´…à´µàµ¼", "S": "à´¸àµ†à´•àµà´•àµ»à´¡àµ"},
}

_saved = []          # (restore callable) in apply order
_installed = False


def _set_attr(obj, name, value):
    had = name in vars(obj)
    old = vars(obj).get(name)
    setattr(obj, name, value)
    _saved.append(lambda: setattr(obj, name, old) if had else delattr(obj, name))


def _set_key(d, key, value):
    had = key in d
    old = d.get(key)
    d[key] = value
    _saved.append(lambda: d.__setitem__(key, old) if had else d.pop(key, None))


def _patch_url(N):
    orig = N._m_url

    def _m_url(self, text):
        # as the vendored _m_url, but a bare host may have sub-domain labels (e-receipt.hdfcbank.com) and may end a
        # sentence ("... irctc.co.in." was left unspelled: the vendored lookahead rejected any following ".")
        pat = (r"https?://[^\s<>]+|www\.[^\s<>]+"
               r"|\b[A-Za-z][\w\-]*(?:\.[A-Za-z0-9][\w\-]*)*\.(?:com|org|net|ai|io|co|dev|app|edu|gov|in)"
               r"(?:\.(?:in|uk|us))?(?:[/?#][^\s<>]*)?(?![\w]|\.\w)")
        for m in re.finditer(pat, text):
            token = m.group(0).rstrip(".,!?;)")
            body = re.sub(r"^https?://", "", token)
            yield m.start(), m.start() + len(token), self._spell_uri_part(body)
        # UPI IDs with a dotted local part (rahul.deshmukh@okaxis) were left raw (the "." unspoken); spell the ID
        for m in re.finditer(r"(?<![\w.@])[A-Za-z][\w\-]*\.[\w.\-]*@[A-Za-z][A-Za-z0-9]{1,15}(?![\w@]|\.\w)", text):
            yield m.start(), m.end(), self._spell_uri_part(m.group(0))

    N._m_url = _m_url
    _saved.append(lambda: setattr(N, "_m_url", orig))


def install(on: bool = True):
    global _installed, AS_ONES
    if on == _installed:
        return
    import sys
    import mithi_tts.text_norm.tn.normalize  # noqa: F401
    mod = sys.modules["mithi_tts.text_norm.tn.normalize"]
    if on:
        L = mod.language_table("as")
        AS_ONES = AS_ONES or _as_ones()
        _set_attr(L, "ONES", AS_ONES)
        _set_attr(L, "ORDINAL_EXCEPTIONS", {1: "à¦ªà§à§°à¦¥à¦®", 2: "à¦¦à§à¦¬à¦¿à¦¤à§€à¦¯à¦¼", 3: "à¦¤à§ƒà¦¤à§€à¦¯à¦¼", 4: "à¦šà¦¤à§à§°à§à¦¥", 5: "à¦ªà¦žà§à¦šà¦®",
                                            6: "à¦·à¦·à§à¦ ", 7: "à¦¸à¦ªà§à¦¤à¦®", 8: "à¦…à¦·à§à¦Ÿà¦®", 9: "à¦¨à§±à¦®", 10: "à¦¦à¦¶à¦®"})
        _set_attr(L, "LETTERS", {**{k: _as_script(v) for k, v in L.LETTERS.items()}, "S": "à¦à¦›", "Y": "à§±à¦¾à¦‡"})
        for k, v in list(L.SYM.items()):
            if _as_script(v) != v:
                _set_key(L.SYM, k, _as_script(v))
        for k, v in _AS_SYM.items():
            _set_key(L.SYM, k, v)
        _set_attr(L, "PERIODS", {"morning": "à¦ªà§à§±à¦¾", "afternoon": "à¦¦à§à¦ªà§°à§€à¦¯à¦¼à¦¾", "evening": "à¦¸à¦¨à§à¦§à¦¿à¦¯à¦¼à¦¾", "night": "à§°à¦¾à¦¤à¦¿"})
        _set_attr(L, "PERIOD_DETECT", {**L.PERIOD_DETECT, **_AS_PERIOD_DETECT})
        _set_attr(L, "TIME_ON_HOUR", "{h} à¦¬à¦œà¦¾")
        # Bengali "à¦Ÿà¦¾" is a counting classifier in Assamese (2023 à¦Ÿà¦¾ à¦…à¦­à¦¿à¦¯à§‹à¦—), not an o'clock marker
        _set_attr(L, "TIME_MARKERS", ["à¦¬à¦¾à¦œà§‡", "à¦¬à¦œà¦¾à¦¤", "à¦¬à¦œà¦¾à§°", "à¦¬à¦œà¦¾à¦²à§ˆ", "à¦¬à¦œà¦¾à¦¤à¦•à§ˆ", "à¦¬à¦œà¦¾"])
        _set_attr(L, "MARKER_SUFFIXES", {"à¦¬à¦œà¦¾à¦¤": "à¦¤", "à¦¬à¦œà¦¾à§°": "à§°", "à¦¬à¦œà¦¾à¦²à§ˆ": "à¦²à§ˆ", "à¦¬à¦œà¦¾à¦¤à¦•à§ˆ": "à¦¤à¦•à§ˆ"})
        _set_attr(L, "POWER", "{base}à§° à¦˜à¦¾à¦¤ {exp}")
        _set_attr(L, "DECADE_SHORT", "{tens}à§° à¦¦à¦¶à¦•")
        _set_attr(L, "DECADE_FULL", "{tens}à§° à¦¦à¦¶à¦•")
        _set_attr(L, "ABBREV_UNITS", {"MINS": "à¦®à¦¿à¦¨à¦¿à¦Ÿ", "MIN": "à¦®à¦¿à¦¨à¦¿à¦Ÿ", "HRS": "à¦˜à¦£à§à¦Ÿà¦¾", "HR": "à¦˜à¦£à§à¦Ÿà¦¾",
                                      "SECS": "à¦›à§‡à¦•à§‡à¦£à§à¦¡", "SEC": "à¦›à§‡à¦•à§‡à¦£à§à¦¡", "KMS": "à¦•à¦¿à¦²à§‹à¦®à¦¿à¦Ÿà¦¾à§°"})
        _set_attr(L, "attach_postposition", types.MethodType(_as_attach, L))
        for code, word in _PER.items():
            sym = mod.language_table(code).SYM
            if "per" not in sym:
                _set_key(sym, "per", word)
        for code, words in _PER_UNITS.items():
            T = mod.language_table(code)
            uw = dict(T.UNIT_WORDS)
            for k, w in words.items():
                uw.setdefault(k, w)
            _set_attr(T, "UNIT_WORDS", uw)
        _patch_url(mod.Normalizer)
    else:
        while _saved:
            _saved.pop()()
    mod._engine.cache_clear()      # regexes are built from the tables
    _installed = on
'''),
    'mithi_tts.text_norm.rules': (False, r'''"""Targeted pre/post rules around the vendored normaliser (mithi_tts.text_norm.tn), derived from the eval rows that still
failed with text normalisation on.

The vendored copy stays byte-identical. Each rule rewrites one input shape that the strict normaliser either rejects
(it then falls back to reading the span character by character: "15-08-25" -> "one five minus zero eight minus two
five") or reads in a way listeners do not expect (a PIN code as "five lakh sixty thousand thirty eight"):

  R1 date_2y      DD-MM-YY, DD/MM/YY, DD.MM.YY            -> four-digit year (pivot: 00-50 -> 20xx, 51-99 -> 19xx),
                                                              then the normaliser's own date reading
  R2 expiry       MM/YY or MM/YYYY next to an expiry / validity / date word or a CVV -> "<month> <year>"
  R3 fy_range     2023-24 (second part = next year)       -> "<year> <two-digit number>"
  R4 digit_groups 4111 2233 4455 6677, 98765 43210, 1800 123 4567 (>= 10 digits in space-separated groups)
                                                           -> one digit run, which the normaliser reads digit by digit
  R5 pin_keyword  PIN / à¤ªà¤¿à¤¨ / ... followed by 4-6 digits    -> digit by digit
  R6 pincode      a bare 6-digit number (no currency or unit next to it, no grouping commas) -> digit by digit
  R7 version      v4.5.2                                  -> "v four point five point two" (language's decimal word)
  R8 ta_date_year Tamil "à®†à®•à®¸à¯à®Ÿà¯ 15, 2025"                  -> "15 à®†à®•à®¸à¯à®Ÿà¯ 2025" (the normaliser keeps the year)
  R10 pct_range   3.5%-4%                                 -> 3.5-4% (the normaliser's own percent range)
  R11 per_price   â‚¹500/kg                                 -> "<amount> <per word> K G"
  R12 per_annum   8.5% p.a.                               -> "P A" (en: "per annum")
  R9 paise        rupee amounts with two decimals read "<n> rupees <m> paise" in every language (the tables of
                  bn en gu kn ml mr pa ta had no paise word, so they read "point five zero rupees")
  post:
  P1 tag_comma    a piece that ends in "," before an inline [tag] keeps its comma (the normaliser turns a trailing
                  comma into a full stop, and the model then ends the sentence at the tag)
  P2 uri_dash     "-" inside an e-mail / URL is spoken in the request language, not as the English word "dash"
  P3 uri_acronym  lowercase acronym labels in an e-mail / URL (sbi, kyc, hdfc in hdfcbank, sbi in onlinesbi) are
                  spelled letter by letter in the request language; other labels keep their English spelling
Language-table corrections (Assamese lexicon, "per" units, sub-domain hosts) live in mithi_tts.text_norm.lexicon.
"""
from __future__ import annotations

import re

__all__ = ["pre", "post", "install", "RULES"]

RULES = ("date_2y", "expiry", "fy_range", "digit_groups", "pin_keyword", "pincode", "version", "ta_date_year",
         "paise", "pct_range", "per_price", "per_annum", "tag_comma", "uri_dash", "uri_acronym")

PAISE = {"en": "paise", "bn": "à¦ªà¦¯à¦¼à¦¸à¦¾", "gu": "àªªà«ˆàª¸àª¾", "kn": "à²ªà³ˆà²¸à³†", "ml": "à´ªàµˆà´¸", "mr": "à¤ªà¥ˆà¤¸à¥‡", "pa": "à¨ªà©ˆà¨¸à©‡",
         "ta": "à®ªà¯ˆà®šà®¾"}            # hi as or te already have one
DASH = {"hi": "à¤¡à¥ˆà¤¶", "mr": "à¤¡à¥…à¤¶", "bn": "à¦¡à§à¦¯à¦¾à¦¶", "as": "à¦¡à§‡à¦¶à§à¦¬", "gu": "àª¡à«…àª¶", "pa": "à¨¡à©ˆà¨¸à¨¼", "or": "à¬¡à­à­Ÿà¬¾à¬¸à­",
        "ta": "à®Ÿà®¾à®·à¯", "te": "à°¡à°¾à°·à±", "kn": "à²¡à³à²¯à²¾à²¶à³", "ml": "à´¡à´¾à´·àµ"}

# expiry / validity / date words (any language; matched case-insensitively in the 40 characters before the span)
_EXPIRY_WORDS = [
    "expiry", "expires", "expire", "exp", "valid", "validity", "thru", "expiration",
    "à¤à¤•à¥à¤¸à¤ªà¤¾à¤¯à¤°à¥€", "à¤¸à¤®à¤¾à¤ªà¥à¤¤à¤¿", "à¤µà¥ˆà¤§à¤¤à¤¾", "à¤®à¥à¤¦à¤¤", "à¤¸à¤®à¤¾à¤ªà¥à¤¤à¥€",
    "à¦®à§‡à¦¯à¦¼à¦¾à¦¦", "à¦®à§‡à¦¯à¦¼à¦¾à¦¦", "à¦à¦•à§à¦¸à¦ªà¦¾à¦¯à¦¼à¦¾à¦°à¦¿", "à¦à¦•à§à¦¸à¦ªà¦¾à¦¯à¦¼à¦¾à¦°à¦¿", "à¦à¦•à§à¦¸à¦ªà¦¾à¦¯à¦¼à¦¾à§°à§€", "à¦®à§à¦¯à¦¾à¦¦",
    "àªàª•à«àª¸àªªàª¾àª¯àª°à«€", "àª¸àª®àª¾àªªà«àª¤àª¿", "àª®àª¾àª¨à«àª¯àª¤àª¾", "à¨®à¨¿à¨†à¨¦", "à¨à¨•à¨¸à¨ªà¨¾à¨‡à¨°à©€",
    "à®•à®¾à®²à®¾à®µà®¤à®¿", "à®Žà®•à¯à®¸à¯à®ªà¯ˆà®°à®¿", "à°—à°¡à±à°µà±", "à°Žà°•à±à°¸à±â€Œà°ªà±ˆà°°à±€", "à°Žà°•à±à°¸à±à°ªà±ˆà°°à±€",
    "à²®à³à²•à³à²¤à²¾à²¯", "à²…à²‚à²¤à²¿à²® à²¦à²¿à²¨à²¾à²‚à²•", "à²Žà²•à³à²¸à³â€Œà²ªà³ˆà²°à²¿", "à´•à´¾à´²à´¾à´µà´§à´¿", "à´Žà´•àµà´¸àµà´ªà´¯à´±à´¿", "à¬®à¬¿à¬†à¬¦", "à¬à¬•à­à¬¸à¬ªà¬¾à­Ÿà¬¾à¬°à­€", "à¬¸à¬®à¬¾à¬ªà­à¬¤à¬¿",
]
_DATE_WORDS = ["date", "à¤¤à¤¾à¤°à¥€à¤–", "à¤¤à¤¿à¤¥à¤¿", "à¤¦à¤¿à¤¨à¤¾à¤‚à¤•", "à¤¡à¥‡à¤Ÿ", "à¦¤à¦¾à¦°à¦¿à¦–", "à¦¤à¦¾à§°à¦¿à¦–", "àª¤àª¾àª°à«€àª–", "àª¡à«‡àªŸ", "à¨¤à¨¾à¨°à©€à¨–à¨¼", "à¨¤à¨¾à¨°à©€à¨–", "à¨¡à©‡à¨Ÿ",
               "à®¤à¯‡à®¤à®¿", "à®Ÿà¯‡à®Ÿà¯", "à°¤à±‡à°¦à±€", "à°¡à±‡à°Ÿà±", "à²¦à²¿à²¨à²¾à²‚à²•", "à²¡à³‡à²Ÿà³", "à´¤àµ€à´¯à´¤à´¿", "à´¡àµ‡à´±àµà´±àµ", "à¬¤à¬¾à¬°à¬¿à¬–", "à¬¡à­‡à¬Ÿà­"]
_alt = lambda ws: "|".join(sorted(map(re.escape, ws), key=len, reverse=True))
_EXPIRY_RE = re.compile(_alt(_EXPIRY_WORDS), re.I)
_DATE_RE = re.compile(_alt(_DATE_WORDS), re.I)
_PIN_WORDS = ["pin", "à¤ªà¤¿à¤¨", "à¤ªà¤¿à¤¨à¤•à¥‹à¤¡", "à²ªà²¿à²¨à³", "à°ªà°¿à°¨à±", "à®ªà®¿à®©à¯", "à´ªà´¿àµ»", "à¦ªà¦¿à¦¨", "à¬ªà¬¿à¬¨", "à¨ªà¨¿à©°à¨¨", "à¨ªà¨¿à¨¨", "àªªàª¿àª¨"]
_PIN_RE = re.compile(r"(?<![\w])(?:" + "|".join(map(re.escape, _PIN_WORDS)) + r")(?:[\s-]*(?:code|à¤•à¥‹à¤¡|à¦•à§‹à¦¡|àª•à«‹àª¡|à¨•à©‹à¨¡|à¬•à­‹à¬¡|à²•à³‹à²¡à³|à°•à±‹à°¡à±|à®•à¯‹à®Ÿà¯|à´•àµ‹à´¡àµ))?"
                     r"(?:\s*(?:no\.?|number|à¤¨à¤‚à¤¬à¤°|à¤¨à¤®à¥à¤¬à¤°))?\s*[:\-]?\s*(\d{4,6})(?![\d,]|\.\d)", re.I)
# words right after a number that make it an amount / quantity (R6 must not read those digit by digit)
_AMOUNT_AFTER = re.compile(
    r"\s*(?:rupees?|rs\.?|inr|dollars?|usd|paise|lakh|crore|thousand|people|users|customers|km|kg|kms|"
    r"à¤°à¥à¤ªà¤¯à¥‡|à¤°à¥à¤ªà¤|à¤°à¥à¤ªà¤¯à¤¾|à¤°à¥\.?|à¤²à¤¾à¤–|à¤•à¤°à¥‹à¤¡à¤¼|à¤¹à¤œà¤¼à¤¾à¤°|à¤¹à¤œà¤¾à¤°|à¤²à¥‹à¤—|à¦²à¦¾à¦–|à¦Ÿà¦¾à¦•à¦¾|à¦Ÿà¦•à¦¾|à¦•à§‹à¦Ÿà¦¿|àª°à«‚àªªàª¿àª¯àª¾|àª²àª¾àª–|à¨°à©à¨ªà¨|à¨²à©±à¨–|à®°à¯‚à®ªà®¾à®¯à¯|à®°à¯‚\.?|à®²à®Ÿà¯à®šà®®à¯|"
    r"à°°à±‚à°ªà°¾à°¯à°²à±|à°°à±‚\.?|à°²à°•à±à°·à°²|à²°à³‚à²ªà²¾à²¯à²¿|à²°à³‚\.?|à²²à²•à³à²·|à´°àµ‚à´ª|à´²à´•àµà´·à´‚|à¬Ÿà¬™à­à¬•à¬¾|à¬²à¬•à­à¬·|à¨•à¨°à©‹à©œ|à¤•à¥‹à¤Ÿà¥€)(?![\w])", re.I)
_CURRENCY_BEFORE = re.compile(r"(?:â‚¹|\$|â‚¬|Â£|\brs\.?|\binr|à¤°à¥\.?|à®°à¯‚\.?|à°°à±‚\.?|à²°à³‚\.?)\s*$", re.I)
_NUMBER_WORD_BEFORE = re.compile(
    r"(?:\bno\.?|\bnumber|\bplot|\bflat|\bhouse|\bdoor|\bh\.\s?no\.?|à¤¨à¤‚à¤¬à¤°|à¤¨à¤®à¥à¤¬à¤°|à¤•à¥à¤°à¤®à¤¾à¤‚à¤•|à¤¨à¤‚\.?|à¦¨à¦‚|à¦¨à¦®à§à¦¬à¦°|à¦¨à¦®à§à¦¬à§°|à¨¨à©°à¨¬à¨°|àª¨àª‚àª¬àª°|à¬¨à¬®à­à¬¬à¬°|"
    r"à®Žà®£à¯|à°¨à°‚à°¬à°°à±|à²¸à²‚à²–à³à²¯à³†|à²¨à²‚à²¬à²°à³|à´¨à´®àµà´ªàµ¼)\s*[:\-]?\s*$", re.I)
_TA_MONTHS = ["à®œà®©à®µà®°à®¿", "à®ªà®¿à®ªà¯à®°à®µà®°à®¿", "à®®à®¾à®°à¯à®šà¯", "à®à®ªà¯à®°à®²à¯", "à®®à¯‡", "à®œà¯‚à®©à¯", "à®œà¯‚à®²à¯ˆ", "à®†à®•à®¸à¯à®Ÿà¯", "à®šà¯†à®ªà¯à®Ÿà®®à¯à®ªà®°à¯", "à®…à®•à¯à®Ÿà¯‹à®ªà®°à¯",
              "à®¨à®µà®®à¯à®ªà®°à¯", "à®Ÿà®¿à®šà®®à¯à®ªà®°à¯"]
_URI_RE = re.compile(r"https?://\S+|www\.\S+|\b[\w.%+\-]+@[\w.\-]+\.[A-Za-z]{2,}|\b[A-Za-z][\w.\-]*@[A-Za-z][A-Za-z0-9]{1,15}\b|\b[A-Za-z][\w\-]*(?:\.[A-Za-z0-9][\w\-]*)*\.(?:com|org|net|ai|io|co|dev|app|edu|gov|in)(?:\.(?:in|uk|us))?(?:[/?#][^\s<>]*)?")

# P3: lowercase acronym labels inside e-mail / URL hosts and paths are spelled letter by letter (sbi.co.in ->
# "S B I dot co dot in", hdfcbank.com -> "H D F C bank dot com"); other labels keep their English spelling.
_URI_ACRONYMS = {"sbi", "hdfc", "icici", "kyc", "irctc", "pnb", "rbi", "idbi", "npci", "uidai", "epfo", "mf", "lic",
                 "upi", "gst", "sms", "atm", "emi", "bob", "boi", "uco", "iob", "nsdl", "cdsl", "pnr", "otp", "ifsc"}
_URI_ACR_PREFIX = ("icici", "irctc", "uidai", "hdfc", "idbi", "npci", "epfo", "sbi", "pnb", "kyc", "rbi")
_URI_ACR_SUFFIX = ("sbi", "mf")

_SPELL_UNITS = {"kg", "km", "gm", "g", "ml", "ltr", "l"}   # R11: read letter by letter (as the normaliser reads "15 kg")

_installed = False


def install(on: bool = True):
    """R9: add the paise word to the language tables that lack it (bn en gu kn ml mr pa ta), so the normaliser's own
    main+sub-unit path reads â‚¹95.41 as "<n> rupees <m> paise" (with its native scale words and case suffixes).
    The tables are module-level; install(False) removes the added words again (used only by rules=False, the
    eval-parity mode; do not mix rules=False with concurrent rules=True calls)."""
    global _installed
    if on == _installed:
        return
    from mithi_tts.text_norm.tn.normalize import language_table
    for code, word in PAISE.items():
        sym = language_table(code).SYM
        if on:
            sym.setdefault("paisa", word)
        elif sym.get("paisa") == word:
            del sym["paisa"]
    _installed = on


def _table(lang):
    from mithi_tts.text_norm.tn.normalize import language_table
    return language_table(lang)


def _tn_id_context(L, before):
    """True when the normaliser itself will read the next number digit by digit (an ID keyword such as PIN / OTP /
    number in the three tokens before it); the rule then leaves the span alone so the normaliser's postposition
    sandhi ("...à²’à²‚à²¦à²°à²²à³à²²à²¿") is kept."""
    from mithi_tts.text_norm.tn import normalize as _N   # the function; its module holds the keyword sets
    import sys
    mod = sys.modules["mithi_tts.text_norm.tn.normalize"]
    kws = mod._ID_KEYWORDS_EN | mod._OTP_INPUT_WORDS | {w.lower() for w in L.ID_KEYWORDS}
    toks = re.findall(r"[\w#\u0900-\u0DFF']+", before.replace("\u200c", "").lower())[-3:]
    return any(t in kws for t in toks)


def _digits(L, s):
    return ", ".join(L.digit(int(c)) for c in s)


def _year4(yy):
    y = int(yy)
    return 2000 + y if y <= 50 else 1900 + y


def _decimal_word(L):
    w = L.decimal(1, "2")
    a, b = L.cardinal(1), L.digit(2)
    return w[len(a):len(w) - len(b)].strip() if w.startswith(a) and w.endswith(b) else "point"


def pre(text: str, lang: str) -> str:
    """Apply R1-R8 outside e-mail / URL spans (those are identifiers, spelled by the normaliser)."""
    try:
        L = _table(lang)
    except Exception:
        return text
    out, pos = [], 0
    for m in _URI_RE.finditer(text):
        out.append(_pre(text[pos:m.start()], lang, L)); out.append(m.group(0)); pos = m.end()
    out.append(_pre(text[pos:], lang, L))
    return "".join(out)


def _pre(text, lang, L):
    t = text

    # R10 percent ranges written with the sign twice: 3.5%-4% (the normaliser read "...percentminus four percent")
    t = re.sub(r"(?<![\d.])(\d+(?:\.\d+)?)\s?%\s?[-â€“]\s?(\d+(?:\.\d+)?)\s?%", r"\1-\2%", t)

    # R11 price per unit: â‚¹500/kg (the "/" was dropped and "kg" left in Latin letters)
    def r11(m):
        unit, per = m.group(2), L.SYM.get("per") or "per"
        if lang != "en" and unit.lower() in _SPELL_UNITS:
            unit = " ".join(L.letter(c) for c in unit.upper())
        return f"{m.group(1)} {per} {unit}"
    t = re.sub(r"((?:[â‚¹$â‚¬Â£]|\b(?:Rs\.?|INR)\s?)\s?\d[\d,]*(?:\.\d+)?)\s?/\s?(kg|km|gm|g|ml|ltr|l|day|month|night|year|person)\b", r11, t,
               flags=re.I)

    # R12 "p.a." (per annum) after a rate was left as the raw token "p.a."
    t = re.sub(r"(?<![\w.])[pP]\.\s?[aA]\.?(?![\w])", lambda m: "per annum" if lang == "en" else f"{L.letter('P')} {L.letter('A')}", t)

    # R7 version strings
    def r7(m):
        parts = m.group(1).split(".")
        dw = _decimal_word(L)
        return f"{L.letter('v')} " + f" {dw} ".join(L.cardinal(int(p)) for p in parts)
    t = re.sub(r"(?<![\w.])[vV](\d+(?:\.\d+){1,3})(?![\w]|\.\d)", r7, t)

    # R4 space-separated digit groups (card / phone numbers): join into one run (>= 10 digits in total)
    def r4(m):
        groups = m.group(0).split()
        return "".join(groups) if sum(len(g) for g in groups) >= 10 else m.group(0)
    t = re.sub(r"(?<![\d,.])(?:\d{4,5}(?: \d{3,5}){1,3}|\d{2,4} \d{3,4} \d{4})(?![\d,]|\.\d)", r4, t)

    # R1 dates with a two-digit year
    def r1(m):
        d, sep, mo, yy = m.group(1), m.group(2), m.group(3), m.group(4)
        if not (1 <= int(d) <= 31 and 1 <= int(mo) <= 12):
            return m.group(0)
        if sep == "-" and len(d) == 1 and len(mo) == 1:          # 8-2-45: a door number, not a date
            return m.group(0)
        if _NUMBER_WORD_BEFORE.search(t[max(0, m.start() - 16):m.start()]):   # "House No. 12-5-24", "à¦˜à§° à¦¨à¦‚ ..."
            return m.group(0)
        return f"{d}{sep}{mo}{sep}{_year4(yy)}"
    t = re.sub(r"(?<![\d/.\-])(\d{1,2})([/.\-])(\d{1,2})\2(\d{2})(?![\d/\-]|\.\d)", r1, t)

    # R2 card expiry / month-year next to an expiry, date or CVV word
    def r2(m):
        mo, y = int(m.group(1)), m.group(2)
        before = t[max(0, m.start() - 40):m.start()]
        after = t[m.end():m.end() + 40]
        strong = _EXPIRY_RE.search(before) or re.search(r"\bcvv\b", before + " " + after, re.I)
        weak = _DATE_RE.search(before) and (len(y) == 4 or int(y) > 12)    # second part > 12: not DD/MM, so MM/YY
        if not (strong or weak):
            return m.group(0)
        year = int(y) if len(y) == 4 else 2000 + int(y)
        return f"{L.month(mo)} {L.year(year)}"
    t = re.sub(r"(?<![\d/.\-])(0?[1-9]|1[0-2])/(\d{2}|20\d{2})(?![\d/]|\.\d)", r2, t)

    # R3 financial-year ranges 2023-24
    def r3(m):
        y, yy = int(m.group(1)), int(m.group(2))
        if (y + 1) % 100 != yy:
            return m.group(0)
        return f"{L.year(y)} {L.cardinal(yy)}"
    t = re.sub(r"(?<![\d\-])((?:19|20)\d{2})[-â€“](\d{2})(?![\d\-])", r3, t)

    # R5 PIN keyword + digits
    def r5(m):
        if _tn_id_context(L, t[max(0, m.start(1) - 60):m.start(1)]):
            return m.group(0)
        return m.group(0)[:m.start(1) - m.start(0)] + _digits(L, m.group(1))
    t = _PIN_RE.sub(r5, t)

    # R6 bare six-digit numbers (Indian PIN codes): digit by digit unless an amount / quantity
    def r6(m):
        before = t[max(0, m.start() - 8):m.start()]
        if (_CURRENCY_BEFORE.search(before) or _AMOUNT_AFTER.match(t, m.end())
                or re.search(r"[A-Za-z0-9]\s?-\s?$|[#Ã·Ã—*/+=<>]\s*$", before)       # ID suffix (CUST-567890), #ref, maths
                or re.match(r"\s*[Ã·Ã—*/+=<>^]", t[m.end():])
                or _tn_id_context(L, re.split(r"[.!?;à¥¤à¥¥\n]", t[max(0, m.start() - 60):m.start()])[-1])):
            return m.group(0)
        return _digits(L, m.group(0))
    t = re.sub(r"(?<![\d,.â‚¹$#\w])[1-9]\d{5}(?![\d,]|\.\d|\s*%)", r6, t)

    # R8 Tamil "Month D, YYYY" -> "D Month YYYY"
    if lang == "ta":
        months = "|".join(_TA_MONTHS)
        t = re.sub(rf"(?<!\w)({months})\s+(\d{{1,2}}),?\s+(\d{{4}})(?!\d)", r"\2 \1 \3", t)
    return t


TAG_COMMA = False   # P1 off: no measured gain (6 vs 6 backstop cuts on the 14 affected emotion rows x2 seeds)


def post(out: str, seg: str, lang: str, before_tag: bool) -> str:
    # P1 keep a trailing comma that precedes an inline [tag] (disabled, see TAG_COMMA)
    if TAG_COMMA and before_tag and seg.rstrip().endswith(",") and out.rstrip().endswith("."):
        out = out.rstrip()[:-1] + ","
    # P2 spoken hyphen inside e-mail / URL in the request language
    uris = [m.group(0) for m in _URI_RE.finditer(seg)]
    if lang in DASH and uris:
        out = re.sub(r"(?<=\s)dash(?=\s)", DASH[lang], out)
    # P3 acronym labels inside e-mail / URL
    labels = {w for u in uris for w in re.findall(r"[a-z]+", re.sub(r"^https?://", "", u))
              if w in _URI_ACRONYMS or w.startswith(_URI_ACR_PREFIX) or w.endswith(_URI_ACR_SUFFIX)}
    if labels:
        try:
            L = _table(lang)
        except Exception:
            return out
        spell = lambda w: " ".join(L.letter(c.upper()) for c in w)

        def p3(m):
            w = m.group(0)
            if w in _URI_ACRONYMS:
                return spell(w)
            for a in _URI_ACR_PREFIX:
                if w.startswith(a) and len(w) - len(a) >= 3:
                    return spell(a) + " " + w[len(a):]
            for a in _URI_ACR_SUFFIX:
                if w.endswith(a) and len(w) - len(a) >= 4:
                    return w[:-len(a)] + " " + spell(a)
            return w
        out = re.sub(r"(?<![\w.@/])(?:" + "|".join(sorted(map(re.escape, labels), key=len, reverse=True)) + r")(?![\w]|[.@/\-]\w)",
                     p3, out)
    return out
'''),
    'mithi_tts.text_norm.tn': (True, r'''"""Vendored text normaliser (upstream commit fe5e6ec).

The modules in this directory are byte-identical copies of the source (see maya3/text/README.md); only this
__init__.py was added so they import as a package.  Use mithi_tts.text_norm.normalise_text rather than this package directly.
"""
from .normalize import NormalizationError, normalize, resolve_language  # noqa: F401
'''),
    'mithi_tts.text_norm.tn.assamese_numbers': (False, r'''"""Assamese number data from AI4Bharat/indic-numtowords (MIT).
Source: https://github.com/AI4Bharat/indic-numtowords/blob/master/indic_numtowords/asm/data/nums.py
License: assamese_numbers.LICENSE. Data only; the upstream parser is not used.
"""
ONES = ['à¦¶à§‚à¦¨à§à¦¯', 'à¦à¦•', 'à¦¦à§à¦‡', 'à¦¤à¦¿à¦¨à¦¿', 'à¦šà¦¾à§°à¦¿', 'à¦ªà¦¾à¦à¦š', 'à¦›à¦¯à¦¼', 'à¦¸à¦¾à¦¤', 'à¦†à¦ ', 'à¦¨', 'à¦¦à¦¹', 'à¦à¦˜à¦¾à§°', 'à¦¬à¦¾à§°', 'à¦¤à§‡à§°', 'à¦šà§ˆà¦§à§à¦¯', 'à¦ªà§‹à¦¨à§à¦§à§°', 'à¦·à§‹à¦²à§à¦²', 'à¦¸à§‹à¦¤à§°', 'à¦“à¦ à§°', 'à¦Šà¦¨à§ˆà¦¶', 'à¦¬à¦¿à¦¶', 'à¦à¦•à§ˆà¦¶', 'à¦¬à¦¾à¦‡à¦¶', 'à¦¤à§‡à¦‡à¦¶', 'à¦šà§Œà¦¬à¦¿à¦¶', 'à¦ªà¦à¦šà¦¿à¦¶', 'à¦›à¦¾à¦¬à§à¦¬à¦¿à¦¶', 'à¦¸à¦¾à¦¤à¦¾à¦‡à¦¶', 'à¦†à¦ à¦¾à¦‡à¦¶', 'à¦Šà¦¨à¦¤à§à¦°à¦¿à¦¶', 'à¦¤à§à¦°à¦¿à¦¶', 'à¦à¦•à¦¤à§à¦°à¦¿à¦¶', 'à¦¬à¦¤à§à¦°à¦¿à¦¶', 'à¦¤à§‡à¦¤à§à¦°à¦¿à¦¶', 'à¦šà§Œà¦¤à§à¦°à¦¿à¦¶', 'à¦ªà¦à¦¯à¦¼à¦¤à§à¦°à¦¿à¦¶', 'à¦›à¦¯à¦¼à¦¤à§à¦°à¦¿à¦¶', 'à¦¸à¦¾à¦¤à¦¤à§à¦°à¦¿à¦¶', 'à¦†à¦ à¦¤à§à¦°à¦¿à¦¶', 'à¦Šà¦¨à¦šà¦²à§à¦²à¦¿à¦¶', 'à¦šà¦²à§à¦²à¦¿à¦¶', 'à¦à¦•à¦šà¦²à§à¦²à¦¿à¦¶', 'à¦¬à¦¿à¦¯à¦¼à¦¾à¦²à§à¦²à¦¿à¦¶', 'à¦¤à¦¿à¦¯à¦¼à¦¾à¦²à§à¦²à¦¿à¦¶', 'à¦šà§Œà¦šà¦²à§à¦²à¦¿à¦¶', 'à¦ªà¦à¦žà§à¦šà¦²à§à¦²à¦¿à¦¶', 'à¦›à¦¯à¦¼à¦šà¦²à§à¦²à¦¿à¦¶', 'à¦¸à¦¾à¦¤à¦šà¦²à§à¦²à¦¿à¦¶', 'à¦†à¦ à¦šà¦²à§à¦²à¦¿à¦¶', 'à¦Šà¦¨à¦ªà¦žà§à¦šà¦¾à¦¶', 'à¦ªà¦žà§à¦šà¦¾à¦¶', 'à¦à¦•à¦¾à§±à¦¨', 'à¦¬à¦¾à§±à¦¨', 'à¦¤à§‡à¦ªà¦¨', 'à¦šà§Œà§±à¦¨', 'à¦ªà¦à¦šà¦ªà¦¨', 'à¦›à¦¾à¦ªà¦¨', 'à¦¸à¦¾à¦¤à¦¾à§±à¦¨', 'à¦†à¦ à¦¾à§±à¦¨', 'à¦Šà¦¨à¦·à¦¾à¦ à¦¿', 'à¦·à¦¾à¦ à¦¿', 'à¦à¦•à¦·à¦·à§à¦ à¦¿', 'à¦¬à¦¾à¦·à¦·à§à¦ à¦¿', 'à¦¤à§‡à¦·à¦·à§à¦ à¦¿', 'à¦šà§Œà¦·à¦·à§à¦ à¦¿', 'à¦ªà¦à¦¯à¦¼à¦·à¦·à§à¦ à¦¿', 'à¦›à¦¯à¦¼à¦·à¦·à§à¦ à¦¿', 'à¦¸à¦¾à¦¤à¦·à¦·à§à¦ à¦¿', 'à¦†à¦ à¦·à¦·à§à¦ à¦¿', 'à¦Šà¦¨à¦¸à¦¤à§à¦¤à§°', 'à¦¸à¦¤à§à¦¤à§°', 'à¦à¦¸à¦¤à§à¦¤à§°', 'à¦¬à¦¾à¦¸à¦¤à§à¦¤à§°', 'à¦¤à§‡à¦¸à¦¤à§à¦¤à§°', 'à¦šà§Œà¦¸à¦¤à§à¦¤à§°', 'à¦ªà¦à¦¯à¦¼à¦¸à¦¤à§à¦¤à§°', 'à¦›à¦¯à¦¼à¦¸à¦¤à§à¦¤à§°', 'à¦¸à¦¾à¦¤à¦¸à¦¤à§à¦¤à§°', 'à¦†à¦ à¦¸à¦¤à§à¦¤à§°', 'à¦Šà¦¨à¦¾à¦¶à§€', 'à¦†à¦¶à§€', 'à¦à¦•à¦¾à¦¶à§€', 'à¦¬à¦¿à§°à¦¾à¦¶à§€', 'à¦¤à¦¿à§°à¦¾à¦¶à§€', 'à¦šà§Œà§°à¦¾à¦¶à§€', 'à¦ªà¦à¦šà¦¾à¦¶à§€', 'à¦›à¦¿à¦¯à¦¼à¦¾à¦¶à§€', 'à¦¸à¦¾à¦¤à¦¾à¦¶à§€', 'à¦†à¦ à¦¾à¦¶à§€', 'à¦Šà¦¨à¦¨à¦¬à§à¦¬à§ˆ', 'à¦¨à¦¬à§à¦¬à§ˆ', 'à¦à¦•à¦¾à¦¨à¦¬à§à¦¬à§ˆ', 'à¦¬à¦¿à§°à¦¾à¦¨à¦¬à§à¦¬à§ˆ', 'à¦¤à¦¿à§°à¦¾à¦¨à¦¬à§à¦¬à§ˆ', 'à¦šà§Œà§°à¦¾à¦¨à¦¬à§à¦¬à§ˆ', 'à¦ªà¦à¦šà¦¾à¦¨à¦¬à§à¦¬à§ˆ', 'à¦›à¦¿à¦¯à¦¼à¦¾à¦¨à¦¬à§à¦¬à§ˆ', 'à¦¸à¦¾à¦¤à¦¾à¦¨à¦¬à§à¦¬à§ˆ', 'à¦†à¦ à¦¾à¦¨à¦¬à§à¦¬à§ˆ', 'à¦¨à¦¿à§°à¦¾à¦¨à¦¬à§à¦¬à§ˆ']
HUNDREDS = ['', 'à¦à¦¶', 'à¦¦à§à¦¶', 'à¦¤à¦¿à¦¨à¦¿à¦¶', 'à¦šà¦¾à§°à¦¿à¦¶', 'à¦ªà¦¾à¦à¦šà¦¶', 'à¦›à¦¯à¦¼à¦¶', 'à¦¸à¦¾à¦¤à¦¶', 'à¦†à¦ à¦¶', 'à¦¨à¦¶']
'''),
    'mithi_tts.text_norm.tn.identifier_rules': (False, r'''"""Opt-in identifier rendering; no model calls or prose transliteration.

Uppercase dictionary components prescribe character spelling. In particular,
AXIS is spelled as an identifier rather than guessed as the word 'access'.
Ordinary words stay words; existing locale letter/symbol tables supply sounds.
"""
import json
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def vocabulary():
    return json.loads(Path(__file__).with_name("identifier_words.json").read_text(encoding="utf-8"))


def spell_uri(part, lang, spell_digits):
    data = vocabulary()

    def word(token, expand=True):
        if expand and token.lower() in data["brands"]:
            return " ".join(word(p, False) for p in data["brands"][token.lower()])
        if (len(token) == 1 or token.isupper() or token.upper() in data["acronyms"]
                or token.upper() in lang.ABBREV_UNITS):
            return " ".join(lang.letter(c) for c in token)
        pieces = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[A-Z]+", token)
        if len(pieces) > 1:
            return " ".join(word(p, False) for p in pieces)
        return token

    symbols = {"@": "at", ".": "dot", "_": "underscore", "-": "minus",
               "/": "slash", "+": "plus", "=": "equals", "&": "amp",
               "#": "hash", "%": "percent"}
    out = []
    # Retain non-ASCII runs intact, including Indic combining marks.
    for token in re.findall(r"[A-Za-z]+|[0-9]+|[@._/\-+?=&#%:~]|[^A-Za-z0-9@._/\-+?=&#%:~]+", part):
        if token.isascii() and token.isalpha():
            out.append(word(token))
        elif token.isdecimal():
            out.append(spell_digits(token, comma=False))
        elif token in symbols:
            out.append(lang.SYM[symbols[token]])
        else:
            out.append({"?": "question mark", ":": "colon", "~": "tilde"}.get(token, token))
    return " ".join(p for p in out if p.strip())


def group_digits(digits, lang, sizes=None):
    """Spaces inside groups, comma pause between groups; every digit survives."""
    if sizes is None:
        if len(digits) == 16:
            sizes = [4] * 4
        elif len(digits) == 10:
            sizes = [3, 3, 4]
        else:
            sizes = [4] * (len(digits) // 4) + [len(digits) % 4]
    out, start = [], 0
    for size in sizes:
        if size:
            out.append(" ".join(lang.digit(int(c)) for c in digits[start:start + size]))
            start += size
    assert start == len(digits)
    return ", ".join(out)


# These shapes identify card/telephone numbers even without English keywords.
# Dates and currency retain higher matcher priority in the normalizer.
LONG_RUN = re.compile(
    r"(?<![A-Za-z0-9.,])(?:\+\d{1,3}[ -]+|91[ -]+)?"
    r"(?:\d{4}(?:[ -]\d{4}){3}|\d{5}[ -]\d{5}|\d{3}[ -]\d{3}[ -]\d{4}|\d{4}[ -]\d{7}|\d{8,})"
    r"(?![A-Za-z0-9]|[.,]\d)"
)


def long_runs(text, lang):
    for match in LONG_RUN.finditer(text):
        token = match.group().lstrip("+")
        groups = re.split(r"[ -]", token)
        digits = "".join(groups)
        sizes = [len(g) for g in groups] if len(groups) > 1 else None
        spoken = group_digits(digits, lang, sizes)
        if match.group().startswith("+"):
            spoken = lang.SYM["plus"] + " " + spoken
        yield match.start(), match.end(), spoken
'''),
    'mithi_tts.text_norm.tn.indic_languages': (False, r'''"""Native language tables recovered from MayaTeam/enterprise.

Source: 7384e91b02848ed791060c2eb368d52c9b92203c:gpu-origin/normalize_all.py
Only LangBase subclasses were extracted; the legacy parser was NOT copied.
The review branch also repairs singular units and Hindi irregular ordinals, and
adds explicit Hindi/Telugu month-spelling aliases for calendar validation.
These are inherited candidate lexicons, not independently native-speaker approved.
See docs/NORMALIZATION_REVIEW.md for provenance, limitations and release gates.
"""
if __package__:
    from .normalize import LangBase
else:
    from normalize import LangBase


class Hindi(LangBase):
    code = "hi"
    name = "Hindi"
    native_digits = "à¥¦à¥§à¥¨à¥©à¥ªà¥«à¥¬à¥­à¥®à¥¯"
    SINGULAR_NOUNS = {"à¤˜à¤‚à¤Ÿà¥‡": "à¤˜à¤‚à¤Ÿà¤¾", "à¤°à¥à¤ªà¤¯à¥‡": "à¤°à¥à¤ªà¤¯à¤¾", "à¤ªà¥ˆà¤¸à¥‡": "à¤ªà¥ˆà¤¸à¤¾"}

    ONES = [
        "à¤¶à¥‚à¤¨à¥à¤¯", "à¤à¤•", "à¤¦à¥‹", "à¤¤à¥€à¤¨", "à¤šà¤¾à¤°", "à¤ªà¤¾à¤à¤š", "à¤›à¤¹", "à¤¸à¤¾à¤¤", "à¤†à¤ ", "à¤¨à¥Œ",
        "à¤¦à¤¸", "à¤—à¥à¤¯à¤¾à¤°à¤¹", "à¤¬à¤¾à¤°à¤¹", "à¤¤à¥‡à¤°à¤¹", "à¤šà¥Œà¤¦à¤¹", "à¤ªà¤‚à¤¦à¥à¤°à¤¹", "à¤¸à¥‹à¤²à¤¹", "à¤¸à¤¤à¥à¤°à¤¹", "à¤…à¤ à¤¾à¤°à¤¹", "à¤‰à¤¨à¥à¤¨à¥€à¤¸",
        "à¤¬à¥€à¤¸", "à¤‡à¤•à¥à¤•à¥€à¤¸", "à¤¬à¤¾à¤ˆà¤¸", "à¤¤à¥‡à¤ˆà¤¸", "à¤šà¥Œà¤¬à¥€à¤¸", "à¤ªà¤šà¥à¤šà¥€à¤¸", "à¤›à¤¬à¥à¤¬à¥€à¤¸", "à¤¸à¤¤à¥à¤¤à¤¾à¤ˆà¤¸", "à¤…à¤Ÿà¥à¤ à¤¾à¤ˆà¤¸", "à¤‰à¤¨à¤¤à¥€à¤¸",
        "à¤¤à¥€à¤¸", "à¤‡à¤•à¤¤à¥€à¤¸", "à¤¬à¤¤à¥à¤¤à¥€à¤¸", "à¤¤à¥ˆà¤‚à¤¤à¥€à¤¸", "à¤šà¥Œà¤‚à¤¤à¥€à¤¸", "à¤ªà¥ˆà¤‚à¤¤à¥€à¤¸", "à¤›à¤¤à¥à¤¤à¥€à¤¸", "à¤¸à¥ˆà¤‚à¤¤à¥€à¤¸", "à¤…à¤¡à¤¼à¤¤à¥€à¤¸", "à¤‰à¤¨à¤¤à¤¾à¤²à¥€à¤¸",
        "à¤šà¤¾à¤²à¥€à¤¸", "à¤‡à¤•à¤¤à¤¾à¤²à¥€à¤¸", "à¤¬à¤¯à¤¾à¤²à¥€à¤¸", "à¤¤à¥ˆà¤‚à¤¤à¤¾à¤²à¥€à¤¸", "à¤šà¥Œà¤µà¤¾à¤²à¥€à¤¸", "à¤ªà¥ˆà¤‚à¤¤à¤¾à¤²à¥€à¤¸", "à¤›à¤¿à¤¯à¤¾à¤²à¥€à¤¸", "à¤¸à¥ˆà¤‚à¤¤à¤¾à¤²à¥€à¤¸", "à¤…à¤¡à¤¼à¤¤à¤¾à¤²à¥€à¤¸", "à¤‰à¤¨à¤šà¤¾à¤¸",
        "à¤ªà¤šà¤¾à¤¸", "à¤‡à¤•à¥à¤¯à¤¾à¤µà¤¨", "à¤¬à¤¾à¤µà¤¨", "à¤¤à¤¿à¤°à¤ªà¤¨", "à¤šà¥Œà¤µà¤¨", "à¤ªà¤šà¤ªà¤¨", "à¤›à¤ªà¥à¤ªà¤¨", "à¤¸à¤¤à¥à¤¤à¤¾à¤µà¤¨", "à¤…à¤Ÿà¥à¤ à¤¾à¤µà¤¨", "à¤‰à¤¨à¤¸à¤ ",
        "à¤¸à¤¾à¤ ", "à¤‡à¤•à¤¸à¤ ", "à¤¬à¤¾à¤¸à¤ ", "à¤¤à¤¿à¤°à¤¸à¤ ", "à¤šà¥Œà¤‚à¤¸à¤ ", "à¤ªà¥ˆà¤‚à¤¸à¤ ", "à¤›à¤¿à¤¯à¤¾à¤¸à¤ ", "à¤¸à¤¡à¤¼à¤¸à¤ ", "à¤…à¤¡à¤¼à¤¸à¤ ", "à¤‰à¤¨à¤¹à¤¤à¥à¤¤à¤°",
        "à¤¸à¤¤à¥à¤¤à¤°", "à¤‡à¤•à¤¹à¤¤à¥à¤¤à¤°", "à¤¬à¤¹à¤¤à¥à¤¤à¤°", "à¤¤à¤¿à¤¹à¤¤à¥à¤¤à¤°", "à¤šà¥Œà¤¹à¤¤à¥à¤¤à¤°", "à¤ªà¤šà¤¹à¤¤à¥à¤¤à¤°", "à¤›à¤¿à¤¹à¤¤à¥à¤¤à¤°", "à¤¸à¤¤à¤¹à¤¤à¥à¤¤à¤°", "à¤…à¤ à¤¹à¤¤à¥à¤¤à¤°", "à¤‰à¤¨à¥à¤¯à¤¾à¤¸à¥€",
        "à¤…à¤¸à¥à¤¸à¥€", "à¤‡à¤•à¥à¤¯à¤¾à¤¸à¥€", "à¤¬à¤¯à¤¾à¤¸à¥€", "à¤¤à¤¿à¤°à¤¾à¤¸à¥€", "à¤šà¥Œà¤°à¤¾à¤¸à¥€", "à¤ªà¤šà¤¾à¤¸à¥€", "à¤›à¤¿à¤¯à¤¾à¤¸à¥€", "à¤¸à¤¤à¥à¤¤à¤¾à¤¸à¥€", "à¤…à¤ à¤¾à¤¸à¥€", "à¤¨à¤µà¤¾à¤¸à¥€",
        "à¤¨à¤¬à¥à¤¬à¥‡", "à¤‡à¤•à¥à¤¯à¤¾à¤¨à¤µà¥‡", "à¤¬à¤¾à¤¨à¤µà¥‡", "à¤¤à¤¿à¤°à¤¾à¤¨à¤µà¥‡", "à¤šà¥Œà¤°à¤¾à¤¨à¤µà¥‡", "à¤ªà¤‚à¤šà¤¾à¤¨à¤µà¥‡", "à¤›à¤¿à¤¯à¤¾à¤¨à¤µà¥‡", "à¤¸à¤¤à¥à¤¤à¤¾à¤¨à¤µà¥‡", "à¤…à¤Ÿà¥à¤ à¤¾à¤¨à¤µà¥‡", "à¤¨à¤¿à¤¨à¥à¤¯à¤¾à¤¨à¤µà¥‡",
    ]
    HUNDRED = "à¤¸à¥Œ"
    THOUSAND = "à¤¹à¤œà¤¼à¤¾à¤°"
    LAKH = "à¤²à¤¾à¤–"
    CRORE = "à¤•à¤°à¥‹à¤¡à¤¼"

    LETTERS = {
        "A": "à¤", "B": "à¤¬à¥€", "C": "à¤¸à¥€", "D": "à¤¡à¥€", "E": "à¤ˆ", "F": "à¤à¤«à¤¼",
        "G": "à¤œà¥€", "H": "à¤à¤š", "I": "à¤†à¤ˆ", "J": "à¤œà¥‡", "K": "à¤•à¥‡", "L": "à¤à¤²",
        "M": "à¤à¤®", "N": "à¤à¤¨", "O": "à¤“", "P": "à¤ªà¥€", "Q": "à¤•à¥à¤¯à¥‚", "R": "à¤†à¤°",
        "S": "à¤à¤¸", "T": "à¤Ÿà¥€", "U": "à¤¯à¥‚", "V": "à¤µà¥€", "W": "à¤¡à¤¬à¥à¤²à¥à¤¯à¥‚",
        "X": "à¤à¤•à¥à¤¸", "Y": "à¤µà¤¾à¤ˆ", "Z": "à¤œà¤¼à¥‡à¤¡",
    }

    MONTHS = ["à¤œà¤¨à¤µà¤°à¥€", "à¤«à¤¼à¤°à¤µà¤°à¥€", "à¤®à¤¾à¤°à¥à¤š", "à¤…à¤ªà¥à¤°à¥ˆà¤²", "à¤®à¤ˆ", "à¤œà¥‚à¤¨",
              "à¤œà¥à¤²à¤¾à¤ˆ", "à¤…à¤—à¤¸à¥à¤¤", "à¤¸à¤¿à¤¤à¤‚à¤¬à¤°", "à¤…à¤•à¥à¤Ÿà¥‚à¤¬à¤°", "à¤¨à¤µà¤‚à¤¬à¤°", "à¤¦à¤¿à¤¸à¤‚à¤¬à¤°"]
    MONTH_ALIASES = {"à¤«à¤°à¤µà¤°à¥€": 2, "à¤¸à¤¿à¤¤à¤®à¥à¤¬à¤°": 9, "à¤¨à¤µà¤®à¥à¤¬à¤°": 11, "à¤¦à¤¿à¤¸à¤®à¥à¤¬à¤°": 12}

    SYM = {
        "rupee": "à¤°à¥à¤ªà¤¯à¥‡", "dollar": "à¤¡à¥‰à¤²à¤°", "euro": "à¤¯à¥‚à¤°à¥‹", "pound": "à¤ªà¤¾à¤‰à¤‚à¤¡",
        "cent": "à¤¸à¥‡à¤‚à¤Ÿ", "paisa": "à¤ªà¥ˆà¤¸à¥‡",
        "percent": "à¤ªà¥à¤°à¤¤à¤¿à¤¶à¤¤",
        "point": "à¤¦à¤¶à¤®à¤²à¤µ",
        "minus": "à¤®à¤¾à¤‡à¤¨à¤¸", "plus": "à¤ªà¥à¤²à¤¸", "equals": "à¤¬à¤°à¤¾à¤¬à¤°",
        "times": "à¤—à¥à¤£à¤¾", "divide": "à¤­à¤¾à¤—",
        "squared": "à¤•à¤¾ à¤µà¤°à¥à¤—", "sqrt": "à¤µà¤°à¥à¤—à¤®à¥‚à¤²",
        "lt": "à¤¸à¥‡ à¤•à¤®", "gt": "à¤¸à¥‡ à¤œà¤¼à¥à¤¯à¤¾à¤¦à¤¾",
        "plus_minus": "à¤§à¤¨-à¤‹à¤£",
        "at": "à¤à¤Ÿ à¤¦ à¤°à¥‡à¤Ÿ", "underscore": "à¤…à¤‚à¤¡à¤°à¤¸à¥à¤•à¥‹à¤°", "dot": "à¤¡à¥‰à¤Ÿ",
        "slash": "à¤¸à¥à¤²à¥ˆà¤¶", "hash": "à¤¹à¥ˆà¤¶", "amp": "à¤”à¤°",
        "half": "à¤†à¤§à¤¾", "quarter": "à¤šà¥Œà¤¥à¤¾à¤ˆ", "three_quarters": "à¤¤à¥€à¤¨ à¤šà¥Œà¤¥à¤¾à¤ˆ",
        "degree": "à¤¡à¤¿à¤—à¥à¤°à¥€", "celsius": "à¤¡à¤¿à¤—à¥à¤°à¥€ à¤¸à¥‡à¤²à¥à¤¸à¤¿à¤¯à¤¸", "fahrenheit": "à¤¡à¤¿à¤—à¥à¤°à¥€ à¤«à¤¼à¤¾à¤°à¥‡à¤¨à¤¹à¤¾à¤‡à¤Ÿ",
        "minute": "à¤®à¤¿à¤¨à¤Ÿ", "second": "à¤¸à¥‡à¤•à¤‚à¤¡", "per": "à¤ªà¥à¤°à¤¤à¤¿",
    }

    TIME_ON_HOUR = "{h} à¤¬à¤œà¥‡"
    TIME_HM = "{h} à¤¬à¤œà¤•à¤° {m} à¤®à¤¿à¤¨à¤Ÿ"
    RANGE = "{a} à¤¸à¥‡ {b}"
    RANGE_TO = "{a} à¤¸à¥‡ {b} à¤¤à¤•"
    RATIO = "{a} à¤…à¤¨à¥à¤ªà¤¾à¤¤ {b}"
    POWER = "{base} à¤•à¥€ à¤˜à¤¾à¤¤ {exp}"
    FRACTION = "{num} à¤¬à¤Ÿà¤¾ {den}"
    DECADE_SHORT = "{tens} à¤•à¥‡"
    DECADE_FULL = "{tens} à¤•à¤¾ à¤¦à¤¶à¤•"

    PERIODS = {"morning": "à¤¸à¥à¤¬à¤¹", "afternoon": "à¤¦à¥‹à¤ªà¤¹à¤°", "evening": "à¤¶à¤¾à¤®", "night": "à¤°à¤¾à¤¤"}

    TIME_MARKERS = ["à¤¬à¤œà¥‡", "à¤¬à¤œà¤•à¤°"]
    PERIOD_DETECT = {"à¤¸à¥à¤¬à¤¹": "morning", "à¤¦à¥‹à¤ªà¤¹à¤°": "afternoon", "à¤¶à¤¾à¤®": "evening",
                     "à¤°à¤¾à¤¤": "night", "à¤¸à¤µà¥‡à¤°à¥‡": "morning"}
    YEAR_WORDS = ["à¤¸à¤¾à¤²", "à¤µà¤°à¥à¤·", "à¤¸à¤¨à¥", "à¤¸à¤¨"]
    ID_KEYWORDS = ["à¤¨à¤‚à¤¬à¤°", "à¤¨à¤®à¥à¤¬à¤°", "à¤–à¤¾à¤¤à¤¾", "à¤–à¤¾à¤¤à¥‡", "à¤•à¥‹à¤¡", "à¤‘à¤°à¥à¤¡à¤°", "à¤«à¤¼à¥‹à¤¨", "à¤«à¥‹à¤¨", "à¤®à¥‹à¤¬à¤¾à¤‡à¤²"]
    LABEL_WORDS = ["à¤…à¤¨à¥à¤šà¥à¤›à¥‡à¤¦", "à¤§à¤¾à¤°à¤¾", "à¤•à¤®à¤°à¤¾", "à¤•à¤•à¥à¤·à¤¾", "à¤µà¤¾à¤°à¥à¤¡", "à¤…à¤§à¥à¤¯à¤¾à¤¯", "à¤­à¤¾à¤—"]
    ID_AFTER_WORDS = ["à¤–à¤¤à¥à¤®", "à¤¡à¤¾à¤¯à¤²", "à¤•à¥‰à¤²", "à¤¸à¤‚à¤ªà¤°à¥à¤•"]
    NUMBER_WORD = "à¤¨à¤‚à¤¬à¤°"
    ABBREV_UNITS = {"MINS": "à¤®à¤¿à¤¨à¤Ÿ", "HRS": "à¤˜à¤‚à¤Ÿà¥‡", "SECS": "à¤¸à¥‡à¤•à¤‚à¤¡",
                    "KMS": "à¤•à¤¿à¤²à¥‹à¤®à¥€à¤Ÿà¤°"}
    MINUTE_WORDS = ["à¤®à¤¿à¤¨à¤Ÿ"]
    SCALE_WORDS = {"à¤•à¤°à¥‹à¤¡à¤¼": "crore", "à¤²à¤¾à¤–": "lakh", "à¤¹à¤œà¤¼à¤¾à¤°": "thousand", "à¤¹à¤œà¤¾à¤°": "thousand",
                   "à¤…à¤°à¤¬": "arab"}
    CURRENCY_WORDS = {"à¤°à¥à¤ªà¤¯à¥‡": "rupee", "à¤°à¥à¤ªà¤": "rupee", "à¤°à¥à¤ªà¤¯à¤¾": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "à¤ªà¤¹à¤²à¤¾", 2: "à¤¦à¥‚à¤¸à¤°à¤¾", 3: "à¤¤à¥€à¤¸à¤°à¤¾", 4: "à¤šà¥Œà¤¥à¤¾", 6: "à¤›à¤ à¤¾"}
    ORDINAL_SUFFIX = "à¤µà¤¾à¤"
    ORDINAL_INPUT_SUFFIXES = ["à¤µà¥€à¤‚", "à¤µà¤¾à¤", "à¤µà¤¾à¤‚", "à¤µà¥‡à¤‚", "à¤µà¥‡"]

    def _ordinal_compose(self, n, src_suffix):
        # 12à¤µà¥€à¤‚ -> à¤¬à¤¾à¤°à¤¹à¤µà¥€à¤‚ : cardinal + preserved suffix
        if src_suffix:
            if n in self.ORDINAL_EXCEPTIONS:
                # Irregular stems retain gender/case: 1à¤µà¥€à¤‚ -> à¤ªà¤¹à¤²à¥€, not à¤à¤•à¤µà¥€à¤‚.
                ending = {"à¤µà¥€à¤‚": "à¥€", "à¤µà¥‡à¤‚": "à¥‡", "à¤µà¥‡": "à¥‡"}.get(src_suffix, "à¤¾")
                return self.ORDINAL_EXCEPTIONS[n][:-1] + ending
            return f"{self.cardinal(n)}{src_suffix}"
        if n in self.ORDINAL_EXCEPTIONS:
            return self.ORDINAL_EXCEPTIONS[n]
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"


class Bengali(LangBase):
    code = "bn"
    name = "Bengali"
    native_digits = "à§¦à§§à§¨à§©à§ªà§«à§¬à§­à§®à§¯"

    ONES = [
        "à¦¶à§‚à¦¨à§à¦¯", "à¦à¦•", "à¦¦à§à¦‡", "à¦¤à¦¿à¦¨", "à¦šà¦¾à¦°", "à¦ªà¦¾à¦à¦š", "à¦›à¦¯à¦¼", "à¦¸à¦¾à¦¤", "à¦†à¦Ÿ", "à¦¨à¦¯à¦¼",
        "à¦¦à¦¶", "à¦à¦—à¦¾à¦°à§‹", "à¦¬à¦¾à¦°à§‹", "à¦¤à§‡à¦°à§‹", "à¦šà§Œà¦¦à§à¦¦", "à¦ªà¦¨à§‡à¦°à§‹", "à¦·à§‹à¦²à§‹", "à¦¸à¦¤à§‡à¦°à§‹", "à¦†à¦ à¦¾à¦°à§‹", "à¦‰à¦¨à¦¿à¦¶",
        "à¦¬à¦¿à¦¶", "à¦à¦•à§à¦¶", "à¦¬à¦¾à¦‡à¦¶", "à¦¤à§‡à¦‡à¦¶", "à¦šà¦¬à§à¦¬à¦¿à¦¶", "à¦ªà¦à¦šà¦¿à¦¶", "à¦›à¦¾à¦¬à§à¦¬à¦¿à¦¶", "à¦¸à¦¾à¦¤à¦¾à¦¶", "à¦†à¦ à¦¾à¦¶", "à¦Šà¦¨à¦¤à§à¦°à¦¿à¦¶",
        "à¦¤à§à¦°à¦¿à¦¶", "à¦à¦•à¦¤à§à¦°à¦¿à¦¶", "à¦¬à¦¤à§à¦°à¦¿à¦¶", "à¦¤à§‡à¦¤à§à¦°à¦¿à¦¶", "à¦šà§Œà¦¤à§à¦°à¦¿à¦¶", "à¦ªà¦à¦¯à¦¼à¦¤à§à¦°à¦¿à¦¶", "à¦›à¦¤à§à¦°à¦¿à¦¶", "à¦¸à¦¾à¦à¦‡à¦¤à§à¦°à¦¿à¦¶", "à¦†à¦Ÿà¦¤à§à¦°à¦¿à¦¶", "à¦Šà¦¨à¦šà¦²à§à¦²à¦¿à¦¶",
        "à¦šà¦²à§à¦²à¦¿à¦¶", "à¦à¦•à¦šà¦²à§à¦²à¦¿à¦¶", "à¦¬à¦¿à¦¯à¦¼à¦¾à¦²à§à¦²à¦¿à¦¶", "à¦¤à§‡à¦¤à¦¾à¦²à§à¦²à¦¿à¦¶", "à¦šà§à¦¯à¦¼à¦¾à¦²à§à¦²à¦¿à¦¶", "à¦ªà¦à¦¯à¦¼à¦¤à¦¾à¦²à§à¦²à¦¿à¦¶", "à¦›à§‡à¦šà¦²à§à¦²à¦¿à¦¶", "à¦¸à¦¾à¦¤à¦šà¦²à§à¦²à¦¿à¦¶", "à¦†à¦Ÿà¦šà¦²à§à¦²à¦¿à¦¶", "à¦Šà¦¨à¦ªà¦žà§à¦šà¦¾à¦¶",
        "à¦ªà¦žà§à¦šà¦¾à¦¶", "à¦à¦•à¦¾à¦¨à§à¦¨", "à¦¬à¦¾à¦¯à¦¼à¦¾à¦¨à§à¦¨", "à¦¤à¦¿à¦ªà§à¦ªà¦¾à¦¨à§à¦¨", "à¦šà§à¦¯à¦¼à¦¾à¦¨à§à¦¨", "à¦ªà¦žà§à¦šà¦¾à¦¨à§à¦¨", "à¦›à¦¾à¦ªà§à¦ªà¦¾à¦¨à§à¦¨", "à¦¸à¦¾à¦¤à¦¾à¦¨à§à¦¨", "à¦†à¦Ÿà¦¾à¦¨à§à¦¨", "à¦Šà¦¨à¦·à¦¾à¦Ÿ",
        "à¦·à¦¾à¦Ÿ", "à¦à¦•à¦·à¦Ÿà§à¦Ÿà¦¿", "à¦¬à¦¾à¦·à¦Ÿà§à¦Ÿà¦¿", "à¦¤à§‡à¦·à¦Ÿà§à¦Ÿà¦¿", "à¦šà§Œà¦·à¦Ÿà§à¦Ÿà¦¿", "à¦ªà¦à¦¯à¦¼à¦·à¦Ÿà§à¦Ÿà¦¿", "à¦›à§‡à¦·à¦Ÿà§à¦Ÿà¦¿", "à¦¸à¦¾à¦¤à¦·à¦Ÿà§à¦Ÿà¦¿", "à¦†à¦Ÿà¦·à¦Ÿà§à¦Ÿà¦¿", "à¦Šà¦¨à¦¸à¦¤à§à¦¤à¦°",
        "à¦¸à¦¤à§à¦¤à¦°", "à¦à¦•à¦¾à¦¤à§à¦¤à¦°", "à¦¬à¦¾à¦¹à¦¾à¦¤à§à¦¤à¦°", "à¦¤à¦¿à¦¯à¦¼à¦¾à¦¤à§à¦¤à¦°", "à¦šà§à¦¯à¦¼à¦¾à¦¤à§à¦¤à¦°", "à¦ªà¦à¦šà¦¾à¦¤à§à¦¤à¦°", "à¦›à¦¿à¦¯à¦¼à¦¾à¦¤à§à¦¤à¦°", "à¦¸à¦¾à¦¤à¦¾à¦¤à§à¦¤à¦°", "à¦†à¦Ÿà¦¾à¦¤à§à¦¤à¦°", "à¦Šà¦¨à¦†à¦¶à¦¿",
        "à¦†à¦¶à¦¿", "à¦à¦•à¦¾à¦¶à¦¿", "à¦¬à¦¿à¦°à¦¾à¦¶à¦¿", "à¦¤à¦¿à¦°à¦¾à¦¶à¦¿", "à¦šà§à¦°à¦¾à¦¶à¦¿", "à¦ªà¦à¦šà¦¾à¦¶à¦¿", "à¦›à¦¿à¦¯à¦¼à¦¾à¦¶à¦¿", "à¦¸à¦¾à¦¤à¦¾à¦¶à¦¿", "à¦†à¦Ÿà¦¾à¦¶à¦¿", "à¦Šà¦¨à¦¨à¦¬à§à¦¬à¦‡",
        "à¦¨à¦¬à§à¦¬à¦‡", "à¦à¦•à¦¾à¦¨à¦¬à§à¦¬à¦‡", "à¦¬à¦¿à¦°à¦¾à¦¨à¦¬à§à¦¬à¦‡", "à¦¤à¦¿à¦°à¦¾à¦¨à¦¬à§à¦¬à¦‡", "à¦šà§à¦°à¦¾à¦¨à¦¬à§à¦¬à¦‡", "à¦ªà¦à¦šà¦¾à¦¨à¦¬à§à¦¬à¦‡", "à¦›à¦¿à¦¯à¦¼à¦¾à¦¨à¦¬à§à¦¬à¦‡", "à¦¸à¦¾à¦¤à¦¾à¦¨à¦¬à§à¦¬à¦‡", "à¦†à¦Ÿà¦¾à¦¨à¦¬à§à¦¬à¦‡", "à¦¨à¦¿à¦°à¦¾à¦¨à¦¬à§à¦¬à¦‡",
    ]
    HUNDRED = "à¦¶à§‹"
    THOUSAND = "à¦¹à¦¾à¦œà¦¾à¦°"
    LAKH = "à¦²à¦¾à¦–"
    CRORE = "à¦•à§‹à¦Ÿà¦¿"

    LETTERS = {
        "A": "à¦", "B": "à¦¬à¦¿", "C": "à¦¸à¦¿", "D": "à¦¡à¦¿", "E": "à¦‡", "F": "à¦à¦«",
        "G": "à¦œà¦¿", "H": "à¦à¦‡à¦š", "I": "à¦†à¦‡", "J": "à¦œà§‡", "K": "à¦•à§‡", "L": "à¦à¦²",
        "M": "à¦à¦®", "N": "à¦à¦¨", "O": "à¦“", "P": "à¦ªà¦¿", "Q": "à¦•à¦¿à¦‰", "R": "à¦†à¦°",
        "S": "à¦à¦¸", "T": "à¦Ÿà¦¿", "U": "à¦‡à¦‰", "V": "à¦­à¦¿", "W": "à¦¡à¦¬à§à¦²à¦¿à¦‰",
        "X": "à¦à¦•à§à¦¸", "Y": "à¦“à¦¯à¦¼à¦¾à¦‡", "Z": "à¦œà§‡à¦¡",
    }

    MONTHS = ["à¦œà¦¾à¦¨à§à¦¯à¦¼à¦¾à¦°à¦¿", "à¦«à§‡à¦¬à§à¦°à§à¦¯à¦¼à¦¾à¦°à¦¿", "à¦®à¦¾à¦°à§à¦š", "à¦à¦ªà§à¦°à¦¿à¦²", "à¦®à§‡", "à¦œà§à¦¨",
              "à¦œà§à¦²à¦¾à¦‡", "à¦†à¦—à¦¸à§à¦Ÿ", "à¦¸à§‡à¦ªà§à¦Ÿà§‡à¦®à§à¦¬à¦°", "à¦…à¦•à§à¦Ÿà§‹à¦¬à¦°", "à¦¨à¦­à§‡à¦®à§à¦¬à¦°", "à¦¡à¦¿à¦¸à§‡à¦®à§à¦¬à¦°"]

    SYM = {
        "rupee": "à¦Ÿà¦¾à¦•à¦¾", "dollar": "à¦¡à¦²à¦¾à¦°", "euro": "à¦‡à¦‰à¦°à§‹", "pound": "à¦ªà¦¾à¦‰à¦¨à§à¦¡",
        "cent": "à¦¸à§‡à¦¨à§à¦Ÿ",
        "percent": "à¦¶à¦¤à¦¾à¦‚à¦¶",
        "point": "à¦¦à¦¶à¦®à¦¿à¦•",
        "minus": "à¦®à¦¾à¦‡à¦¨à¦¾à¦¸", "plus": "à¦ªà§à¦²à¦¾à¦¸", "equals": "à¦¸à¦®à¦¾à¦¨",
        "times": "à¦—à§à¦£", "divide": "à¦­à¦¾à¦—",
        "squared": "à¦à¦° à¦¬à¦°à§à¦—", "sqrt": "à¦¬à¦°à§à¦—à¦®à§‚à¦²",
        "lt": "à¦à¦° à¦šà§‡à¦¯à¦¼à§‡ à¦•à¦®", "gt": "à¦à¦° à¦šà§‡à¦¯à¦¼à§‡ à¦¬à§‡à¦¶à¦¿",
        "plus_minus": "à¦ªà§à¦²à¦¾à¦¸-à¦®à¦¾à¦‡à¦¨à¦¾à¦¸",
        "at": "à¦…à§à¦¯à¦¾à¦Ÿ à¦¦à§à¦¯ à¦°à§‡à¦Ÿ", "underscore": "à¦†à¦¨à§à¦¡à¦¾à¦°à¦¸à§à¦•à§‹à¦°", "dot": "à¦¡à¦Ÿ",
        "slash": "à¦¸à§à¦²à§à¦¯à¦¾à¦¶", "hash": "à¦¹à§à¦¯à¦¾à¦¶", "amp": "à¦…à§à¦¯à¦¾à¦¨à§à¦¡",
        "half": "à¦…à¦°à§à¦§à§‡à¦•", "quarter": "à¦à¦• à¦šà¦¤à§à¦°à§à¦¥à¦¾à¦‚à¦¶", "three_quarters": "à¦¤à¦¿à¦¨ à¦šà¦¤à§à¦°à§à¦¥à¦¾à¦‚à¦¶",
        "degree": "à¦¡à¦¿à¦—à§à¦°à¦¿", "celsius": "à¦¡à¦¿à¦—à§à¦°à¦¿ à¦¸à§‡à¦²à¦¸à¦¿à¦¯à¦¼à¦¾à¦¸", "fahrenheit": "à¦¡à¦¿à¦—à§à¦°à¦¿ à¦«à¦¾à¦°à§‡à¦¨à¦¹à¦¾à¦‡à¦Ÿ",
        "minute": "à¦®à¦¿à¦¨à¦¿à¦Ÿ", "second": "à¦¸à§‡à¦•à§‡à¦¨à§à¦¡",
    }

    TIME_ON_HOUR = "{h}à¦Ÿà¦¾"
    TIME_HM = "{h}à¦Ÿà¦¾ à¦¬à§‡à¦œà§‡ {m} à¦®à¦¿à¦¨à¦¿à¦Ÿ"
    RANGE = "{a} à¦¥à§‡à¦•à§‡ {b}"
    RANGE_TO = "{a} à¦¥à§‡à¦•à§‡ {b} à¦ªà¦°à§à¦¯à¦¨à§à¦¤"
    RATIO = "{a} à¦…à¦¨à§à¦ªà¦¾à¦¤ {b}"
    POWER = "{base} à¦à¦° à¦˜à¦¾à¦¤ {exp}"
    FRACTION = "{num} à¦¬à¦¾à¦‡ {den}"
    DECADE_SHORT = "{tens} à¦à¦° à¦¦à¦¶à¦•"
    DECADE_FULL = "{tens} à¦à¦° à¦¦à¦¶à¦•"

    PERIODS = {"morning": "à¦¸à¦•à¦¾à¦²", "afternoon": "à¦¦à§à¦ªà§à¦°",
               "evening": "à¦¸à¦¨à§à¦§à§à¦¯à¦¾", "night": "à¦°à¦¾à¦¤"}

    TIME_MARKERS = ["à¦Ÿà¦¾", "à¦Ÿà¦¾à¦¯à¦¼", "à¦Ÿà¦¾à¦°", "à¦¬à§‡à¦œà§‡", "à¦¬à¦¾à¦œà§‡", "à¦¬à¦¾à¦œà¦¬à§‡"]
    PERIOD_DETECT = {"à¦¸à¦•à¦¾à¦²": "morning", "à¦¸à¦•à¦¾à¦²à§‡": "morning",
                     "à¦­à§‹à¦°": "morning", "à¦­à§‹à¦°à§‡": "morning",
                     "à¦¦à§à¦ªà§à¦°": "afternoon", "à¦¦à§à¦ªà§à¦°à§‡": "afternoon",
                     "à¦¬à¦¿à¦•à§‡à¦²": "evening", "à¦¬à¦¿à¦•à§‡à¦²à§‡": "evening", "à¦¬à¦¿à¦•à¦¾à¦²": "evening",
                     "à¦¸à¦¨à§à¦§à§à¦¯à¦¾": "evening", "à¦¸à¦¨à§à¦§à§à¦¯à¦¾à¦¯à¦¼": "evening",
                     "à¦°à¦¾à¦¤": "night", "à¦°à¦¾à¦¤à§‡": "night"}
    YEAR_WORDS = ["à¦¸à¦¾à¦²", "à¦¸à¦¾à¦²à§‡", "à¦¸à¦¾à¦²à§‡à¦°", "à¦¸à¦¨", "à¦¬à¦›à¦°", "à¦¬à¦›à¦°à§‡"]
    ID_KEYWORDS = ["à¦¨à¦®à§à¦¬à¦°", "à¦¨à¦¾à¦®à§à¦¬à¦¾à¦°", "à¦…à§à¦¯à¦¾à¦•à¦¾à¦‰à¦¨à§à¦Ÿ", "à¦à¦•à¦¾à¦‰à¦¨à§à¦Ÿ", "à¦•à§‹à¦¡", "à¦…à¦°à§à¦¡à¦¾à¦°", "à¦«à§‹à¦¨", "à¦®à§‹à¦¬à¦¾à¦‡à¦²", "à¦“à¦Ÿà¦¿à¦ªà¦¿", "à¦ªà¦¿à¦¨", "à¦†à¦‡à¦¡à¦¿", "à¦ªà¦¾à¦¸à¦“à¦¯à¦¼à¦¾à¦°à§à¦¡", "à¦Ÿà§à¦°à§‡à¦¨", "à¦à¦•à§à¦¸à¦ªà§à¦°à§‡à¦¸", "à¦ªà¦¿à¦à¦¨à¦†à¦°", "à¦•à¦¾à¦°à§à¦¡", "à¦°à§‡à¦«à¦¾à¦°à§‡à¦¨à§à¦¸", "à¦Ÿà§à¦°à¦¾à¦¨à¦œà§‡à¦•à¦¶à¦¨", "à¦Ÿà¦¿à¦•à¦¿à¦Ÿ", "à¦…à§à¦¯à¦¾à¦•à¦¾à¦‰à¦¨à§à¦Ÿà§‡à¦°", "à¦¨à¦®à§à¦¬à¦°à§‡", "à¦†à¦‡à¦¡à¦¿à¦¤à§‡", "à¦‡à¦‰à¦œà¦¾à¦°", "à¦‡à¦¨à¦¸à§à¦Ÿà¦²à¦®à§‡à¦¨à§à¦Ÿ", "à¦•à¦¿à¦¸à§à¦¤à¦¿", "à¦¡à¦¿à¦œà¦¿à¦Ÿ", "à¦…à¦™à§à¦•", "à¦²à§‡à¦¨à¦¦à§‡à¦¨"]
    ID_AFTER_WORDS = ["à¦¡à¦¾à¦¯à¦¼à¦¾à¦²", "à¦•à¦²", "à¦¯à§‹à¦—à¦¾à¦¯à§‹à¦—"]
    NUMBER_WORD = "à¦¨à¦®à§à¦¬à¦°"
    ABBREV_UNITS = {"MINS":"à¦®à¦¿à¦¨à¦¿à¦Ÿ", "MIN":"à¦®à¦¿à¦¨à¦¿à¦Ÿ", "HRS":"à¦˜à¦£à§à¦Ÿà¦¾", "HR":"à¦˜à¦£à§à¦Ÿà¦¾", "SECS":"à¦¸à§‡à¦•à§‡à¦¨à§à¦¡", "SEC":"à¦¸à§‡à¦•à§‡à¦¨à§à¦¡", "KMS":"à¦•à¦¿à¦²à§‹à¦®à¦¿à¦Ÿà¦¾à¦°"}
    LABEL_WORDS = ["à¦…à¦¨à§à¦šà§à¦›à§‡à¦¦", "à¦§à¦¾à¦°à¦¾", "à¦°à§à¦®", "à¦˜à¦°", "à¦•à¦•à§à¦·", "à¦“à¦¯à¦¼à¦¾à¦°à§à¦¡",
                   "à¦…à¦§à§à¦¯à¦¾à¦¯à¦¼", "à¦—à§‡à¦Ÿ", "à¦•à¦¾à¦‰à¦¨à§à¦Ÿà¦¾à¦°"]
    MINUTE_WORDS = ["à¦®à¦¿à¦¨à¦¿à¦Ÿ"]
    SCALE_WORDS = {"à¦•à§‹à¦Ÿà¦¿": "crore", "à¦²à¦¾à¦–": "lakh", "à¦²à¦•à§à¦·": "lakh",
                   "à¦¹à¦¾à¦œà¦¾à¦°": "thousand"}
    CURRENCY_WORDS = {"à¦Ÿà¦¾à¦•à¦¾": "rupee", "à¦Ÿà¦¾à¦•à¦¾à¦°": "rupee", "à¦Ÿà¦¾à¦•à¦¾à¦¯à¦¼": "rupee",
                      "à¦°à§à¦ªà¦¿": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "à¦ªà§à¦°à¦¥à¦®", 2: "à¦¦à§à¦¬à¦¿à¦¤à§€à¦¯à¦¼", 3: "à¦¤à§ƒà¦¤à§€à¦¯à¦¼", 4: "à¦šà¦¤à§à¦°à§à¦¥",
                          5: "à¦ªà¦žà§à¦šà¦®", 6: "à¦·à¦·à§à¦ ", 7: "à¦¸à¦ªà§à¦¤à¦®", 8: "à¦…à¦·à§à¦Ÿà¦®",
                          9: "à¦¨à¦¬à¦®", 10: "à¦¦à¦¶à¦®"}
    ORDINAL_SUFFIX = "à¦¤à¦®"
    ORDINAL_INPUT_SUFFIXES = ["à¦¤à¦®", "à¦²à¦¾", "à¦°à¦¾", "à¦ à¦¾", "à¦‡", "à¦¶à§‡"]

    # Bengali date-of-month ordinals for à§§à¦²à¦¾ / à§¨à¦°à¦¾ / à§©à¦°à¦¾ / à§ªà¦ à¦¾
    _DATE_ORDINALS = {1: "à¦ªà¦¹à§‡à¦²à¦¾", 2: "à¦¦à§‹à¦¸à¦°à¦¾", 3: "à¦¤à§‡à¦¸à¦°à¦¾", 4: "à¦šà§Œà¦ à¦¾"}

    # ------------------------------------------------------------------
    # Hundreds fuse: à¦à¦•+à¦¶à§‹ -> à¦à¦•à¦¶à§‹, à¦šà¦¾à¦°+à¦¶à§‹ -> à¦šà¦¾à¦°à¦¶à§‹ (written solid).
    # ------------------------------------------------------------------
    def hundreds(self, h, rest):
        out = f"{self.ONES[h]}{self.HUNDRED}"
        if rest:
            out += f" {self.two_digit(rest)}"
        return out

    def year(self, y):
        """1947 -> à¦‰à¦¨à¦¿à¦¶à¦¶à§‹ à¦¸à¦¾à¦¤à¦šà¦²à§à¦²à¦¿à¦¶ (fused à¦¶à§‹); 2026 -> à¦¦à§à¦‡ à¦¹à¦¾à¦œà¦¾à¦° à¦›à¦¾à¦¬à§à¦¬à¦¿à¦¶."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            out = f"{self.two_digit(hi)}{self.HUNDRED}"
            if lo:
                out += f" {self.two_digit(lo)}"
            return out
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # Date-style ordinals: à§§à¦²à¦¾ -> à¦ªà¦¹à§‡à¦²à¦¾, à§¨à¦°à¦¾ -> à¦¦à§‹à¦¸à¦°à¦¾, à§ªà¦ à¦¾ -> à¦šà§Œà¦ à¦¾,
        # à§«à¦‡ -> à¦ªà¦¾à¦à¦šà¦‡, à§§à§¯à¦¶à§‡ -> à¦‰à¦¨à¦¿à¦¶à§‡. General: à§§à§¨à¦¤à¦® -> à¦¬à¦¾à¦°à§‹à¦¤à¦®.
        if src_suffix in ("à¦²à¦¾", "à¦°à¦¾", "à¦ à¦¾") and n in self._DATE_ORDINALS:
            return self._DATE_ORDINALS[n]
        if src_suffix == "à¦‡":
            return f"{self.cardinal(n)}à¦‡"
        if src_suffix == "à¦¶à§‡":
            return f"{self.cardinal(n)}à§‡"
        if src_suffix == "à¦¤à¦®":
            return f"{self.cardinal(n)}à¦¤à¦®"
        if n in self.ORDINAL_EXCEPTIONS:
            return self.ORDINAL_EXCEPTIONS[n]
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"

    # postpositions attach to noun-words with Bengali sandhi
    # (à¦Ÿà¦¾à¦•à¦¾+à¦à¦°->à¦Ÿà¦¾à¦•à¦¾à¦°, à¦¶à¦¤à¦¾à¦‚à¦¶+à¦->à¦¶à¦¤à¦¾à¦‚à¦¶à§‡, à¦Ÿà¦¾+à¦®à¦§à§à¦¯à§‡->à¦Ÿà¦¾à¦° à¦®à¦§à§à¦¯à§‡)
    _ATTACH_NOUNS = {"à¦Ÿà¦¾à¦•à¦¾", "à¦ªà¦¯à¦¼à¦¸à¦¾", "à¦¡à¦²à¦¾à¦°", "à¦¶à¦¤à¦¾à¦‚à¦¶", "à¦•à¦¿à¦²à§‹à¦®à¦¿à¦Ÿà¦¾à¦°",
                     "à¦•à¦¿à¦²à§‹à¦—à§à¦°à¦¾à¦®", "à¦®à¦¿à¦Ÿà¦¾à¦°", "à¦²à¦¿à¦Ÿà¦¾à¦°", "à¦—à§à¦°à¦¾à¦®", "à¦¡à¦¿à¦—à§à¦°à¦¿",
                     "à¦®à¦¿à¦¨à¦¿à¦Ÿ", "à¦¸à§‡à¦•à§‡à¦¨à§à¦¡", "à¦Ÿà¦¾", "à¦Ÿà§‹"}

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if last not in self._ATTACH_NOUNS:
            return None
        if t == last:
            return rep                          # duplicate marker
        if t in ("à¦", "à¦¤à§‡", "à¦¯à¦¼"):               # locative
            if last.endswith(("à¦¾","à§‹","à§‡")):
                return rep + "à¦¯à¦¼"                # à¦Ÿà¦¾à¦•à¦¾à¦¯à¦¼, à¦à¦—à¦¾à¦°à§‹à¦Ÿà¦¾à¦¯à¦¼
            return rep + "à§‡"                     # à¦¶à¦¤à¦¾à¦‚à¦¶à§‡, à¦®à¦¿à¦¨à¦¿à¦Ÿà§‡
        if t in ("à¦à¦°", "à¦°"):                     # genitive
            if last.endswith(("à¦¾","à§‹","à§‡")):
                return rep + "à¦°"                 # à¦Ÿà¦¾à¦•à¦¾à¦°, à¦Ÿà¦¾à¦°
            return rep + "à§‡à¦°"                    # à¦®à¦¿à¦¨à¦¿à¦Ÿà§‡à¦°
        if t in ("à¦®à¦§à§à¦¯à§‡", "à¦¨à¦¾à¦—à¦¾à¦¦", "à¦¥à§‡à¦•à§‡", "à¦ªà¦°à§à¦¯à¦¨à§à¦¤", "à¦¦à¦¿à¦¯à¦¼à§‡"):
            g = rep + ("à¦°" if last.endswith(("à¦¾","à§‹","à§‡")) else "à§‡à¦°")
            return g + " " + t                   # à¦à¦—à¦¾à¦°à§‹à¦Ÿà¦¾à¦° à¦®à¦§à§à¦¯à§‡
        return None


class Gujarati(LangBase):
    code = "gu"
    name = "Gujarati"
    native_digits = "à«¦à«§à«¨à«©à«ªà««à«¬à«­à«®à«¯"

    ONES = [
        "àª¶à«‚àª¨à«àª¯", "àªàª•", "àª¬à«‡", "àª¤à«àª°àª£", "àªšàª¾àª°", "àªªàª¾àª‚àªš", "àª›", "àª¸àª¾àª¤", "àª†àª ", "àª¨àªµ",
        "àª¦àª¸", "àª…àª—àª¿àª¯àª¾àª°", "àª¬àª¾àª°", "àª¤à«‡àª°", "àªšà«Œàª¦", "àªªàª‚àª¦àª°", "àª¸à«‹àª³", "àª¸àª¤à«àª¤àª°", "àª…àª¢àª¾àª°", "àª“àª—àª£à«€àª¸",
        "àªµà«€àª¸", "àªàª•àªµà«€àª¸", "àª¬àª¾àªµà«€àª¸", "àª¤à«àª°à«‡àªµà«€àª¸", "àªšà«‹àªµà«€àª¸", "àªªàªšà«àªšà«€àª¸", "àª›àªµà«àªµà«€àª¸", "àª¸àª¤à«àª¤àª¾àªµà«€àª¸", "àª…àª à«àª àª¾àªµà«€àª¸", "àª“àª—àª£àª¤à«àª°à«€àª¸",
        "àª¤à«àª°à«€àª¸", "àªàª•àª¤à«àª°à«€àª¸", "àª¬àª¤à«àª°à«€àª¸", "àª¤à«‡àª¤à«àª°à«€àª¸", "àªšà«‹àª¤à«àª°à«€àª¸", "àªªàª¾àª‚àª¤à«àª°à«€àª¸", "àª›àª¤à«àª°à«€àª¸", "àª¸àª¾àª¡àª¤à«àª°à«€àª¸", "àª†àª¡àª¤à«àª°à«€àª¸", "àª“àª—àª£àªšàª¾àª³à«€àª¸",
        "àªšàª¾àª³à«€àª¸", "àªàª•àª¤àª¾àª³à«€àª¸", "àª¬à«‡àª¤àª¾àª³à«€àª¸", "àª¤à«‡àª‚àª¤àª¾àª³à«€àª¸", "àªšà«àª‚àª®àª¾àª³à«€àª¸", "àªªàª¿àª¸à«àª¤àª¾àª³à«€àª¸", "àª›à«‡àª‚àª¤àª¾àª³à«€àª¸", "àª¸à«àª¡àª¤àª¾àª³à«€àª¸", "àª…àª¡àª¤àª¾àª³à«€àª¸", "àª“àª—àª£àªªàªšàª¾àª¸",
        "àªªàªšàª¾àª¸", "àªàª•àª¾àªµàª¨", "àª¬àª¾àªµàª¨", "àª¤à«àª°à«‡àªªàª¨", "àªšà«‹àªªàª¨", "àªªàª‚àªšàª¾àªµàª¨", "àª›àªªà«àªªàª¨", "àª¸àª¤à«àª¤àª¾àªµàª¨", "àª…àª à«àª àª¾àªµàª¨", "àª“àª—àª£àª¸àª¾àª ",
        "àª¸àª¾àª ", "àªàª•àª¸àª ", "àª¬àª¾àª¸àª ", "àª¤à«àª°à«‡àª¸àª ", "àªšà«‹àª¸àª ", "àªªàª¾àª‚àª¸àª ", "àª›àª¾àª¸àª ", "àª¸àª¡àª¸àª ", "àª…àª¡àª¸àª ", "àª…àª—àª£à«‹àª¸àª¿àª¤à«àª¤à«‡àª°",
        "àª¸àª¿àª¤à«àª¤à«‡àª°", "àªàª•à«‹àª¤à«‡àª°", "àª¬à«‹àª¤à«‡àª°", "àª¤à«‹àª‚àª¤à«‡àª°", "àªšà«àª‚àª®à«‹àª¤à«‡àª°", "àªªàª‚àªšà«‹àª¤à«‡àª°", "àª›à«‹àª¤à«‡àª°", "àª¸àª¿àª¤à«àª¤à«‹àª¤à«‡àª°", "àª…àª à«àª à«‹àª¤à«‡àª°", "àª“àª—àª£àªàª‚àª¸à«€",
        "àªàª‚àª¸à«€", "àªàª•à«àª¯àª¾àª¸à«€", "àª¬à«àª¯àª¾àª¸à«€", "àª¤à«àª¯àª¾àª¸à«€", "àªšà«‹àª°àª¾àª¸à«€", "àªªàª‚àªšàª¾àª¸à«€", "àª›àª¯àª¾àª¸à«€", "àª¸àª¤à«àª¯àª¾àª¸à«€", "àª…àª àª¯àª¾àª¸à«€", "àª¨à«‡àªµà«àª¯àª¾àª¸à«€",
        "àª¨à«‡àªµà«àª‚", "àªàª•àª¾àª£à«àª‚", "àª¬àª¾àª£à«àª‚", "àª¤à«àª°àª¾àª£à«àª‚", "àªšà«‹àª°àª¾àª£à«àª‚", "àªªàª‚àªšàª¾àª£à«àª‚", "àª›àª¨à«àª¨à«àª‚", "àª¸àª¤à«àª¤àª¾àª£à«àª‚", "àª…àª à«àª àª¾àª£à«àª‚", "àª¨àªµà«àªµàª¾àª£à«àª‚",
    ]
    HUNDRED = "àª¸à«‹"
    THOUSAND = "àª¹àªœàª¾àª°"
    LAKH = "àª²àª¾àª–"
    CRORE = "àª•àª°à«‹àª¡"

    # Gujarati hundreds are fused compounds (200 = àª¬àª¸à«‹, not "àª¬à«‡ àª¸à«‹").
    HUNDREDS_FUSED = {
        1: "àªàª•àª¸à«‹", 2: "àª¬àª¸à«‹", 3: "àª¤à«àª°àª£àª¸à«‹", 4: "àªšàª¾àª°àª¸à«‹", 5: "àªªàª¾àª‚àªšàª¸à«‹",
        6: "àª›àª¸à«‹", 7: "àª¸àª¾àª¤àª¸à«‹", 8: "àª†àª àª¸à«‹", 9: "àª¨àªµàª¸à«‹",
    }

    LETTERS = {
        "A": "àª", "B": "àª¬à«€", "C": "àª¸à«€", "D": "àª¡à«€", "E": "àªˆ", "F": "àªàª«",
        "G": "àªœà«€", "H": "àªàªš", "I": "àª†àªˆ", "J": "àªœà«‡", "K": "àª•à«‡", "L": "àªàª²",
        "M": "àªàª®", "N": "àªàª¨", "O": "àª“", "P": "àªªà«€", "Q": "àª•à«àª¯à«", "R": "àª†àª°",
        "S": "àªàª¸", "T": "àªŸà«€", "U": "àª¯à«", "V": "àªµà«€", "W": "àª¡àª¬àª²à«àª¯à«",
        "X": "àªàª•à«àª¸", "Y": "àªµàª¾àª¯", "Z": "àªà«‡àª¡",
    }

    MONTHS = ["àªœàª¾àª¨à«àª¯à«àª†àª°à«€", "àª«à«‡àª¬à«àª°à«àª†àª°à«€", "àª®àª¾àª°à«àªš", "àªàªªà«àª°àª¿àª²", "àª®à«‡", "àªœà«‚àª¨",
              "àªœà«àª²àª¾àªˆ", "àª“àª—àª¸à«àªŸ", "àª¸àªªà«àªŸà«‡àª®à«àª¬àª°", "àª“àª•à«àªŸà«‹àª¬àª°", "àª¨àªµà«‡àª®à«àª¬àª°", "àª¡àª¿àª¸à«‡àª®à«àª¬àª°"]

    SYM = {
        "rupee": "àª°à«‚àªªàª¿àª¯àª¾", "dollar": "àª¡à«‹àª²àª°", "euro": "àª¯à«àª°à«‹", "pound": "àªªàª¾àª‰àª¨à«àª¡",
        "cent": "àª¸à«‡àª¨à«àªŸ",
        "percent": "àªŸàª•àª¾",
        "point": "àª¦àª¶àª¾àª‚àª¶",
        "minus": "àª®àª¾àª‡àª¨àª¸", "plus": "àªªà«àª²àª¸", "equals": "àª¬àª°àª¾àª¬àª°",
        "times": "àª—à«àª£à«àª¯àª¾", "divide": "àª­àª¾àª—à«àª¯àª¾",
        "squared": "àª¨à«‹ àªµàª°à«àª—", "sqrt": "àªµàª°à«àª—àª®à«‚àª³",
        "lt": "àª¥à«€ àª“àª›à«àª‚", "gt": "àª¥à«€ àªµàª§àª¾àª°à«‡",
        "plus_minus": "àªµàª¤à«àª¤àª¾-àª“àª›àª¾",
        "at": "àªàªŸ àª§ àª°à«‡àªŸ", "underscore": "àª…àª¨à«àª¡àª°àª¸à«àª•à«‹àª°", "dot": "àª¡à«‹àªŸ",
        "slash": "àª¸à«àª²à«‡àª¶", "hash": "àª¹à«‡àª¶", "amp": "àª…àª¨à«‡",
        "half": "àª…àª¡àª§à«‹", "quarter": "àªªàª¾", "three_quarters": "àªªà«‹àª£à«‹",
        "degree": "àª¡àª¿àª—à«àª°à«€", "celsius": "àª¡àª¿àª—à«àª°à«€ àª¸à«‡àª²à«àª¸àª¿àª¯àª¸", "fahrenheit": "àª¡àª¿àª—à«àª°à«€ àª«à«‡àª°àª¨àª¹à«€àªŸ",
        "minute": "àª®àª¿àª¨àª¿àªŸ", "second": "àª¸à«‡àª•àª¨à«àª¡",
    }

    TIME_ON_HOUR = "{h} àªµàª¾àª—à«àª¯à«‡"
    TIME_HM = "{h} àªµàª¾àª—à«€àª¨à«‡ {m} àª®àª¿àª¨àª¿àªŸ"
    RANGE = "{a} àª¥à«€ {b}"
    RANGE_TO = "{a} àª¥à«€ {b} àª¸à«àª§à«€"
    RATIO = "{a} àª—à«àª£à«‹àª¤à«àª¤àª° {b}"
    POWER = "{base} àª¨à«€ àª˜àª¾àª¤ {exp}"
    FRACTION = "{num} àª¬àªŸàª¾ {den}"
    DECADE_SHORT = "{tens} àª¨àª¾"
    DECADE_FULL = "{tens} àª¨à«‹ àª¦àª¾àª¯àª•à«‹"

    PERIODS = {"morning": "àª¸àªµàª¾àª°à«‡", "afternoon": "àª¬àªªà«‹àª°à«‡",
               "evening": "àª¸àª¾àª‚àªœà«‡", "night": "àª°àª¾àª¤à«àª°à«‡"}

    TIME_MARKERS = ["àªµàª¾àª—à«àª¯à«‡", "àªµàª¾àª—à«‡", "àªµàª¾àª—à«àª¯àª¾", "àªµàª¾àª—à«àª¯à«àª‚", "àªµàª¾àª—à«€àª¨à«‡", "àª•àª²àª¾àª•à«‡"]
    PERIOD_DETECT = {"àª¸àªµàª¾àª°à«‡": "morning", "àª¸àªµàª¾àª°": "morning", "àª¸àªµàª¾àª°àª¨àª¾": "morning",
                     "àª¬àªªà«‹àª°à«‡": "afternoon", "àª¬àªªà«‹àª°": "afternoon", "àª¬àªªà«‹àª°àª¨àª¾": "afternoon",
                     "àª¸àª¾àª‚àªœà«‡": "evening", "àª¸àª¾àª‚àªœ": "evening", "àª¸àª¾àª‚àªœàª¨àª¾": "evening",
                     "àª°àª¾àª¤à«àª°à«‡": "night", "àª°àª¾àª¤à«‡": "night", "àª°àª¾àª¤": "night", "àª°àª¾àª¤àª¨àª¾": "night"}
    YEAR_WORDS = ["àªµàª°à«àª·", "àªµàª°à«àª·à«‡", "àª¸àª¾àª²", "àª¸àª¨", "àª¸àª¨à«‡"]
    ID_KEYWORDS = ["àª¨àª‚àª¬àª°", "àª¨àª®à«àª¬àª°", "àª–àª¾àª¤àª¾", "àª–àª¾àª¤à«àª‚", "àª–àª¾àª¤àª¾àª‚", "àª•à«‹àª¡", "àª“àª°à«àª¡àª°", "àª«à«‹àª¨", "àª®à«‹àª¬àª¾àª‡àª²", "àª®à«‹àª¬àª¾àªˆàª²", "àªàª•àª¾àª‰àª¨à«àªŸ", "àªàª•àª¾àª‰àª¨à«àªŸàª®àª¾àª‚", "àªàª•àª¾àª‰àª¨à«àªŸàª¨à«‹", "àªªàª¿àª¨", "àª†àªˆàª¡à«€", "àªªàª¾àª¸àªµàª°à«àª¡", "àªŸà«àª°à«‡àª¨", "àªàª•à«àª¸àªªà«àª°à«‡àª¸", "àªªà«€àªàª¨àª†àª°", "àª•àª¾àª°à«àª¡", "àª°à«‡àª«àª°àª¨à«àª¸", "àªŸà«àª°àª¾àª¨à«àªà«‡àª•à«àª¶àª¨", "àªŸàª¿àª•àª¿àªŸ", "àª“àªŸà«€àªªà«€", "àª¨àª‚àª¬àª°àª®àª¾àª‚", "àª¯à«àªàª°", "àª¹àªªà«àª¤à«‹", "àª…àª‚àª•", "àª¡àª¿àªœàª¿àªŸ"]
    ID_AFTER_WORDS = ["àª¡àª¾àª¯àª²", "àª•à«‰àª²", "àª¸àª‚àªªàª°à«àª•"]
    NUMBER_WORD = "àª¨àª‚àª¬àª°"
    ABBREV_UNITS = {"MINS":"àª®àª¿àª¨àª¿àªŸ", "MIN":"àª®àª¿àª¨àª¿àªŸ", "HRS":"àª•àª²àª¾àª•", "HR":"àª•àª²àª¾àª•", "SECS":"àª¸à«‡àª•àª¨à«àª¡", "SEC":"àª¸à«‡àª•àª¨à«àª¡", "KMS":"àª•àª¿àª²à«‹àª®à«€àªŸàª°"}
    LABEL_WORDS = ["àª…àª¨à«àªšà«àª›à«‡àª¦", "àª•àª²àª®", "àª“àª°àª¡à«‹", "àª°à«‚àª®", "àªµà«‹àª°à«àª¡", "àª§à«‹àª°àª£",
                   "àªªà«àª°àª•àª°àª£", "àª­àª¾àª—", "àªµàª°à«àª—"]
    MINUTE_WORDS = ["àª®àª¿àª¨àª¿àªŸ", "àª®àª¿àª¨à«€àªŸ"]
    SCALE_WORDS = {"àª•àª°à«‹àª¡": "crore", "àª²àª¾àª–": "lakh", "àª¹àªœàª¾àª°": "thousand",
                   "àª…àª¬àªœ": "arab"}
    CURRENCY_WORDS = {"àª°à«‚àªªàª¿àª¯àª¾": "rupee", "àª°à«‚àªªàª¿àª¯à«‹": "rupee", "àª°à«‚àªªàª¿àª¯à«‡": "rupee",
                      "àª°à«‚": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "àªªàª¹à«‡àª²à«‹", 2: "àª¬à«€àªœà«‹", 3: "àª¤à«àª°à«€àªœà«‹", 4: "àªšà«‹àª¥à«‹", 6: "àª›àª à«àª à«‹"}
    # Stems for gender/case agreement with a source suffix (1àª²à«€ -> àªªàª¹à«‡àª²à«€).
    ORDINAL_STEMS = {1: "àªªàª¹à«‡àª²", 2: "àª¬à«€àªœ", 3: "àª¤à«àª°à«€àªœ", 4: "àªšà«‹àª¥", 6: "àª›àª à«àª "}
    ORDINAL_SUFFIX = "àª®à«‹"
    ORDINAL_INPUT_SUFFIXES = ["àª®à«€", "àª®à«‹", "àª®à«àª‚", "àª®àª¾", "àª®à«‡",
                              "àª²à«€", "àª²à«‹", "àª²à«àª‚", "àª²àª¾",
                              "àªœà«€", "àªœà«‹", "àªœà«àª‚", "àªœàª¾",
                              "àª¥à«€", "àª¥à«‹", "àª¥à«àª‚", "àª¥àª¾"]

    # ------------------------------------------------------------------
    # Gujarati-specific composition
    # ------------------------------------------------------------------

    def hundreds(self, h, rest):
        """Fused hundreds: 250 -> 'àª¬àª¸à«‹ àªªàªšàª¾àª¸', 147 -> 'àªàª•àª¸à«‹ àª¸à«àª¡àª¤àª¾àª³à«€àª¸'."""
        out = self.HUNDREDS_FUSED[h]
        if rest:
            out += f" {self.two_digit(rest)}"
        return out

    def year(self, y):
        """1100-1999: fused hundred style â€” 1947 -> 'àª“àª—àª£à«€àª¸à«‹ àª¸à«àª¡àª¤àª¾àª³à«€àª¸',
        1857 -> 'àª…àª¢àª¾àª°àª¸à«‹ àª¸àª¤à«àª¤àª¾àªµàª¨'. A trailing àª¸ merges with àª¸à«‹
        (àª“àª—àª£à«€àª¸ + àª¸à«‹ -> àª“àª—àª£à«€àª¸à«‹). Everything else: full cardinal."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            w = self.two_digit(hi)
            w = w[:-1] + self.HUNDRED if w.endswith("àª¸") else w + self.HUNDRED
            if lo:
                return f"{w} {self.two_digit(lo)}"
            return w
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # 1-4, 6 have suppletive stems: 1àª²à«€ -> àªªàª¹à«‡àª²à«€, 2àªœà«€ -> àª¬à«€àªœà«€, 4àª¥àª¾ -> àªšà«‹àª¥àª¾.
        if n in self.ORDINAL_STEMS:
            if src_suffix and len(src_suffix) > 1:
                return self.ORDINAL_STEMS[n] + src_suffix[1:]
            return self.ORDINAL_EXCEPTIONS[n]
        # Regular: cardinal + àª®-suffix (12àª®à«€ -> àª¬àª¾àª°àª®à«€, 5àª®à«‹ -> àªªàª¾àª‚àªšàª®à«‹).
        if src_suffix and src_suffix.startswith("àª®"):
            return f"{self.cardinal(n)}{src_suffix}"
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"

    # Gujarati postpositions agglutinate directly onto the preceding word
    # (àª°à«‚àªªàª¿àª¯àª¾ + àª®àª¾àª‚ -> àª°à«‚àªªàª¿àª¯àª¾àª®àª¾àª‚). Oblique of the plural currency word before
    # a genitive: àª°à«‚àªªàª¿àª¯àª¾ stays (already the citation form here).
    _PP = {"àª®àª¾àª‚", "àª®àª¾àª‚àª¥à«€", "àª¨à«àª‚", "àª¨à«‹", "àª¨à«€", "àª¨àª¾", "àª¨à«‡", "àª¥à«€", "àª¸à«àª§à«€",
           "àªªàª°", "àªªàª°àª¥à«€", "àªµàª¡à«‡", "àª•àª°àª¤àª¾àª‚", "àªœà«‡àªŸàª²àª¾", "àªœà«‡àªŸàª²à«àª‚"}

    # attach postpositions only to noun-words we insert (currency/unit/
    # percent), never to a bare cardinal (Terminal àª¬à«‡ + àªªàª° stays "àª¬à«‡ àªªàª°")
    _ATTACH_NOUNS = {"àª°à«‚àªªàª¿àª¯àª¾", "àªªà«ˆàª¸àª¾", "àª¡à«‹àª²àª°", "àªŸàª•àª¾", "àªŸàª•àª¾àªµàª¾àª°à«€",
                     "àª•àª¿àª²à«‹àª®à«€àªŸàª°", "àª•àª¿àª²à«‹àª—à«àª°àª¾àª®", "àª®à«€àªŸàª°", "àª²àª¿àªŸàª°", "àª—à«àª°àª¾àª®",
                     "àª¡àª¿àª—à«àª°à«€", "àª•àª²àª¾àª•", "àª®àª¿àª¨àª¿àªŸ", "àª¸à«‡àª•àª¨à«àª¡"}

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if t == last and last in self._ATTACH_NOUNS:
            return rep
        if t in self._PP and last in self._ATTACH_NOUNS:
            return rep + t
        return None


class Kannada(LangBase):
    code = "kn"
    name = "Kannada"
    native_digits = "à³¦à³§à³¨à³©à³ªà³«à³¬à³­à³®à³¯"

    ONES = [
        "à²¸à³Šà²¨à³à²¨à³†", "à²’à²‚à²¦à³", "à²Žà²°à²¡à³", "à²®à³‚à²°à³", "à²¨à²¾à²²à³à²•à³", "à²à²¦à³", "à²†à²°à³", "à²à²³à³", "à²Žà²‚à²Ÿà³", "à²’à²‚à²¬à²¤à³à²¤à³",
        "à²¹à²¤à³à²¤à³", "à²¹à²¨à³à²¨à³Šà²‚à²¦à³", "à²¹à²¨à³à²¨à³†à²°à²¡à³", "à²¹à²¦à²¿à²®à³‚à²°à³", "à²¹à²¦à²¿à²¨à²¾à²²à³à²•à³", "à²¹à²¦à²¿à²¨à³ˆà²¦à³", "à²¹à²¦à²¿à²¨à²¾à²°à³", "à²¹à²¦à²¿à²¨à³‡à²³à³", "à²¹à²¦à²¿à²¨à³†à²‚à²Ÿà³", "à²¹à²¤à³à²¤à³Šà²‚à²¬à²¤à³à²¤à³",
        "à²‡à²ªà³à²ªà²¤à³à²¤à³", "à²‡à²ªà³à²ªà²¤à³à²¤à³Šà²‚à²¦à³", "à²‡à²ªà³à²ªà²¤à³à²¤à³†à²°à²¡à³", "à²‡à²ªà³à²ªà²¤à³à²¤à²®à³‚à²°à³", "à²‡à²ªà³à²ªà²¤à³à²¤à²¨à²¾à²²à³à²•à³", "à²‡à²ªà³à²ªà²¤à³à²¤à³ˆà²¦à³", "à²‡à²ªà³à²ªà²¤à³à²¤à²¾à²°à³", "à²‡à²ªà³à²ªà²¤à³à²¤à³‡à²³à³", "à²‡à²ªà³à²ªà²¤à³à²¤à³†à²‚à²Ÿà³", "à²‡à²ªà³à²ªà²¤à³à²¤à³Šà²‚à²¬à²¤à³à²¤à³",
        "à²®à³‚à²µà²¤à³à²¤à³", "à²®à³‚à²µà²¤à³à²¤à³Šà²‚à²¦à³", "à²®à³‚à²µà²¤à³à²¤à³†à²°à²¡à³", "à²®à³‚à²µà²¤à³à²¤à²®à³‚à²°à³", "à²®à³‚à²µà²¤à³à²¤à²¨à²¾à²²à³à²•à³", "à²®à³‚à²µà²¤à³à²¤à³ˆà²¦à³", "à²®à³‚à²µà²¤à³à²¤à²¾à²°à³", "à²®à³‚à²µà²¤à³à²¤à³‡à²³à³", "à²®à³‚à²µà²¤à³à²¤à³†à²‚à²Ÿà³", "à²®à³‚à²µà²¤à³à²¤à³Šà²‚à²¬à²¤à³à²¤à³",
        "à²¨à²²à²µà²¤à³à²¤à³", "à²¨à²²à²µà²¤à³à²¤à³Šà²‚à²¦à³", "à²¨à²²à²µà²¤à³à²¤à³†à²°à²¡à³", "à²¨à²²à²µà²¤à³à²¤à²®à³‚à²°à³", "à²¨à²²à²µà²¤à³à²¤à²¨à²¾à²²à³à²•à³", "à²¨à²²à²µà²¤à³à²¤à³ˆà²¦à³", "à²¨à²²à²µà²¤à³à²¤à²¾à²°à³", "à²¨à²²à²µà²¤à³à²¤à³‡à²³à³", "à²¨à²²à²µà²¤à³à²¤à³†à²‚à²Ÿà³", "à²¨à²²à²µà²¤à³à²¤à³Šà²‚à²¬à²¤à³à²¤à³",
        "à²à²µà²¤à³à²¤à³", "à²à²µà²¤à³à²¤à³Šà²‚à²¦à³", "à²à²µà²¤à³à²¤à³†à²°à²¡à³", "à²à²µà²¤à³à²¤à²®à³‚à²°à³", "à²à²µà²¤à³à²¤à²¨à²¾à²²à³à²•à³", "à²à²µà²¤à³à²¤à³ˆà²¦à³", "à²à²µà²¤à³à²¤à²¾à²°à³", "à²à²µà²¤à³à²¤à³‡à²³à³", "à²à²µà²¤à³à²¤à³†à²‚à²Ÿà³", "à²à²µà²¤à³à²¤à³Šà²‚à²¬à²¤à³à²¤à³",
        "à²…à²°à²µà²¤à³à²¤à³", "à²…à²°à²µà²¤à³à²¤à³Šà²‚à²¦à³", "à²…à²°à²µà²¤à³à²¤à³†à²°à²¡à³", "à²…à²°à²µà²¤à³à²¤à²®à³‚à²°à³", "à²…à²°à²µà²¤à³à²¤à²¨à²¾à²²à³à²•à³", "à²…à²°à²µà²¤à³à²¤à³ˆà²¦à³", "à²…à²°à²µà²¤à³à²¤à²¾à²°à³", "à²…à²°à²µà²¤à³à²¤à³‡à²³à³", "à²…à²°à²µà²¤à³à²¤à³†à²‚à²Ÿà³", "à²…à²°à²µà²¤à³à²¤à³Šà²‚à²¬à²¤à³à²¤à³",
        "à²Žà²ªà³à²ªà²¤à³à²¤à³", "à²Žà²ªà³à²ªà²¤à³à²¤à³Šà²‚à²¦à³", "à²Žà²ªà³à²ªà²¤à³à²¤à³†à²°à²¡à³", "à²Žà²ªà³à²ªà²¤à³à²¤à²®à³‚à²°à³", "à²Žà²ªà³à²ªà²¤à³à²¤à²¨à²¾à²²à³à²•à³", "à²Žà²ªà³à²ªà²¤à³à²¤à³ˆà²¦à³", "à²Žà²ªà³à²ªà²¤à³à²¤à²¾à²°à³", "à²Žà²ªà³à²ªà²¤à³à²¤à³‡à²³à³", "à²Žà²ªà³à²ªà²¤à³à²¤à³†à²‚à²Ÿà³", "à²Žà²ªà³à²ªà²¤à³à²¤à³Šà²‚à²¬à²¤à³à²¤à³",
        "à²Žà²‚à²¬à²¤à³à²¤à³", "à²Žà²‚à²¬à²¤à³à²¤à³Šà²‚à²¦à³", "à²Žà²‚à²¬à²¤à³à²¤à³†à²°à²¡à³", "à²Žà²‚à²¬à²¤à³à²¤à²®à³‚à²°à³", "à²Žà²‚à²¬à²¤à³à²¤à²¨à²¾à²²à³à²•à³", "à²Žà²‚à²¬à²¤à³à²¤à³ˆà²¦à³", "à²Žà²‚à²¬à²¤à³à²¤à²¾à²°à³", "à²Žà²‚à²¬à²¤à³à²¤à³‡à²³à³", "à²Žà²‚à²¬à²¤à³à²¤à³†à²‚à²Ÿà³", "à²Žà²‚à²¬à²¤à³à²¤à³Šà²‚à²¬à²¤à³à²¤à³",
        "à²¤à³Šà²‚à²¬à²¤à³à²¤à³", "à²¤à³Šà²‚à²¬à²¤à³à²¤à³Šà²‚à²¦à³", "à²¤à³Šà²‚à²¬à²¤à³à²¤à³†à²°à²¡à³", "à²¤à³Šà²‚à²¬à²¤à³à²¤à²®à³‚à²°à³", "à²¤à³Šà²‚à²¬à²¤à³à²¤à²¨à²¾à²²à³à²•à³", "à²¤à³Šà²‚à²¬à²¤à³à²¤à³ˆà²¦à³", "à²¤à³Šà²‚à²¬à²¤à³à²¤à²¾à²°à³", "à²¤à³Šà²‚à²¬à²¤à³à²¤à³‡à²³à³", "à²¤à³Šà²‚à²¬à²¤à³à²¤à³†à²‚à²Ÿà³", "à²¤à³Šà²‚à²¬à²¤à³à²¤à³Šà²‚à²¬à²¤à³à²¤à³",
    ]
    HUNDRED = "à²¨à³‚à²°à³"
    THOUSAND = "à²¸à²¾à²µà²¿à²°"
    LAKH = "à²²à²•à³à²·"
    CRORE = "à²•à³‹à²Ÿà²¿"

    # Fused hundreds (standalone) and their combining (oblique) forms used
    # when a 0-99 remainder follows: 147 -> à²¨à³‚à²° à²¨à²²à²µà²¤à³à²¤à³‡à²³à³.
    HUNDREDS = ["", "à²¨à³‚à²°à³", "à²‡à²¨à³à²¨à³‚à²°à³", "à²®à³à²¨à³à²¨à³‚à²°à³", "à²¨à²¾à²¨à³‚à²°à³", "à²à²¨à³‚à²°à³",
                "à²†à²°à³à²¨à³‚à²°à³", "à²à²³à³à²¨à³‚à²°à³", "à²Žà²‚à²Ÿà³à²¨à³‚à²°à³", "à²’à²‚à²¬à³ˆà²¨à³‚à²°à³"]
    HUNDREDS_COMB = ["", "à²¨à³‚à²°", "à²‡à²¨à³à²¨à³‚à²°", "à²®à³à²¨à³à²¨à³‚à²°", "à²¨à²¾à²¨à³‚à²°", "à²à²¨à³‚à²°",
                     "à²†à²°à³à²¨à³‚à²°", "à²à²³à³à²¨à³‚à²°", "à²Žà²‚à²Ÿà³à²¨à³‚à²°", "à²’à²‚à²¬à³ˆà²¨à³‚à²°"]
    THOUSAND_COMB = "à²¸à²¾à²µà²¿à²°à²¦"
    LAKH_COMB = "à²²à²•à³à²·à²¦"

    LETTERS = {
        "A": "à²Ž", "B": "à²¬à²¿", "C": "à²¸à²¿", "D": "à²¡à²¿", "E": "à²‡", "F": "à²Žà²«à³",
        "G": "à²œà²¿", "H": "à²Žà²šà³", "I": "à²", "J": "à²œà³†", "K": "à²•à³†", "L": "à²Žà²²à³",
        "M": "à²Žà²‚", "N": "à²Žà²¨à³", "O": "à²’", "P": "à²ªà²¿", "Q": "à²•à³à²¯à³‚", "R": "à²†à²°à³",
        "S": "à²Žà²¸à³", "T": "à²Ÿà²¿", "U": "à²¯à³", "V": "à²µà²¿", "W": "à²¡à²¬à³à²²à³à²¯à³‚",
        "X": "à²Žà²•à³à²¸à³", "Y": "à²µà³ˆ", "Z": "à²à²¡à³",
    }

    MONTHS = ["à²œà²¨à²µà²°à²¿", "à²«à³†à²¬à³à²°à²µà²°à²¿", "à²®à²¾à²°à³à²šà³", "à²à²ªà³à²°à²¿à²²à³", "à²®à³‡", "à²œà³‚à²¨à³",
              "à²œà³à²²à³ˆ", "à²†à²—à²¸à³à²Ÿà³", "à²¸à³†à²ªà³à²Ÿà³†à²‚à²¬à²°à³", "à²…à²•à³à²Ÿà³‹à²¬à²°à³", "à²¨à²µà³†à²‚à²¬à²°à³", "à²¡à²¿à²¸à³†à²‚à²¬à²°à³"]

    SYM = {
        "rupee": "à²°à³‚à²ªà²¾à²¯à²¿", "dollar": "à²¡à²¾à²²à²°à³", "euro": "à²¯à³‚à²°à³‹", "pound": "à²ªà³Œà²‚à²¡à³",
        "cent": "à²¸à³†à²‚à²Ÿà³",
        "percent": "à²ªà³à²°à²¤à²¿à²¶à²¤",
        "point": "à²¦à²¶à²®à²¾à²‚à²¶",
        "minus": "à²®à³ˆà²¨à²¸à³", "plus": "à²ªà³à²²à²¸à³", "equals": "à²¸à²®",
        "times": "à²—à³à²£à²¿à²¸à³", "divide": "à²­à²¾à²—à²¿à²¸à³",
        "squared": "à²µà²°à³à²—", "sqrt": "à²µà²°à³à²—à²®à³‚à²²",
        "lt": "à²—à²¿à²‚à²¤ à²•à²¡à²¿à²®à³†", "gt": "à²—à²¿à²‚à²¤ à²¹à³†à²šà³à²šà³",
        "plus_minus": "à²ªà³à²²à²¸à³ à²®à³ˆà²¨à²¸à³",
        "at": "à²…à²Ÿà³ à²¦à²¿ à²°à³‡à²Ÿà³", "underscore": "à²…à²‚à²¡à²°à³â€Œà²¸à³à²•à³‹à²°à³", "dot": "à²¡à²¾à²Ÿà³",
        "slash": "à²¸à³à²²à³à²¯à²¾à²¶à³", "hash": "à²¹à³à²¯à²¾à²¶à³", "amp": "à²®à²¤à³à²¤à³",
        "half": "à²…à²°à³à²§", "quarter": "à²•à²¾à²²à³", "three_quarters": "à²®à³à²•à³à²•à²¾à²²à³",
        "degree": "à²¡à²¿à²—à³à²°à²¿", "celsius": "à²¡à²¿à²—à³à²°à²¿ à²¸à³†à²²à³à²¸à²¿à²¯à²¸à³", "fahrenheit": "à²¡à²¿à²—à³à²°à²¿ à²«à³à²¯à²¾à²°à²¨à³â€Œà²¹à³€à²Ÿà³",
        "minute": "à²¨à²¿à²®à²¿à²·", "second": "à²¸à³†à²•à³†à²‚à²¡à³",
    }

    TIME_ON_HOUR = "{h} à²—à²‚à²Ÿà³†"
    TIME_HM = "{h} à²—à²‚à²Ÿà³† {m} à²¨à²¿à²®à²¿à²·"
    RANGE = "{a} à²°à²¿à²‚à²¦ {b}"
    RANGE_TO = "{a} à²°à²¿à²‚à²¦ {b} à²µà²°à³†à²—à³†"
    RATIO = "{a} à²…à²¨à³à²ªà²¾à²¤ {b}"
    POWER = "{base} à²˜à²¾à²¤ {exp}"
    FRACTION = "{num} à²¬à³ˆ {den}"
    DECADE_SHORT = "{tens}à²°"        # decade() passes the à³-stripped stem
    DECADE_FULL = "{tens}à²° à²¦à²¶à²•"

    PERIODS = {"morning": "à²¬à³†à²³à²¿à²—à³à²—à³†", "afternoon": "à²®à²§à³à²¯à²¾à²¹à³à²¨",
               "evening": "à²¸à²‚à²œà³†", "night": "à²°à²¾à²¤à³à²°à²¿"}

    TIME_MARKERS = ["à²—à²‚à²Ÿà³†à²—à³†", "à²—à²‚à²Ÿà³†à²¯", "à²—à²‚à²Ÿà³†"]
    MARKER_SUFFIXES = {"à²—à²‚à²Ÿà³†à²—à³†": "à²—à³†"}
    PERIOD_DETECT = {"à²¬à³†à²³à²¿à²—à³à²—à³†": "morning", "à²¬à³†à²³à²—à³à²—à³†": "morning",
                     "à²®à³à²‚à²œà²¾à²¨à³†": "morning", "à²®à²§à³à²¯à²¾à²¹à³à²¨": "afternoon",
                     "à²¸à²‚à²œà³†": "evening", "à²°à²¾à²¤à³à²°à²¿": "night"}
    YEAR_WORDS = ["à²µà²°à³à²·", "à²‡à²¸à²µà²¿", "à²¸à²¾à²²à²¿à²¨", "à²¸à²¾à²²à³", "à²¸à²¨à³"]
    ID_KEYWORDS = ["à²¨à²‚à²¬à²°à³", "à²¸à²‚à²–à³à²¯à³†", "à²–à²¾à²¤à³†", "à²•à³‹à²¡à³", "à²†à²°à³à²¡à²°à³", "à²«à³‹à²¨à³", "à²®à³Šà²¬à³ˆà²²à³", "à²à²¡à²¿", "à²ªà²¿à²¨à³", "à²ªà²¾à²¸à³â€Œà²µà²°à³à²¡à³", "à²°à³ˆà²²à³", "à²Žà²•à³à²¸à³â€Œà²ªà³à²°à³†à²¸à³", "à²ªà²¿à²Žà²¨à³â€Œà²†à²°à³", "à²•à²¾à²°à³à²¡à³", "à²°à³†à²«à²°à³†à²¨à³à²¸à³", "à²Ÿà³à²°à²¾à²¨à³à²¸à²¾à²•à³à²·à²¨à³", "à²Ÿà²¿à²•à³†à²Ÿà³", "à²“à²Ÿà²¿à²ªà²¿", "à²…à²•à³Œà²‚à²Ÿà³", "à²…à²‚à²•à²¿", "à²¬à²³à²•à³†à²¦à²¾à²°", "à²•à²‚à²¤à³", "à²µà²¹à²¿à²µà²¾à²Ÿà³"]
    ID_AFTER_WORDS = ["à²¡à²¯à²²à³", "à²•à²°à³†", "à²¸à²‚à²ªà²°à³à²•"]
    NUMBER_WORD = "à²¨à²‚à²¬à²°à³"
    ABBREV_UNITS = {"MINS":"à²¨à²¿à²®à²¿à²·", "MIN":"à²¨à²¿à²®à²¿à²·", "HRS":"à²—à²‚à²Ÿà³†", "HR":"à²—à²‚à²Ÿà³†", "SECS":"à²¸à³†à²•à³†à²‚à²¡à³", "SEC":"à²¸à³†à²•à³†à²‚à²¡à³", "KMS":"à²•à²¿à²²à³‹à²®à³€à²Ÿà²°à³"}
    LABEL_WORDS = ["à²µà²¿à²§à²¿", "à²•à²²à²‚", "à²•à³Šà² à²¡à²¿", "à²µà²¾à²°à³à²¡à³", "à²…à²§à³à²¯à²¾à²¯", "à²­à²¾à²—", "à²¤à²°à²—à²¤à²¿"]
    MINUTE_WORDS = ["à²¨à²¿à²®à²¿à²·"]
    SCALE_WORDS = {"à²•à³‹à²Ÿà²¿": "crore", "à²²à²•à³à²·": "lakh", "à²²à²•à³à²·à²¦": "lakh",
                   "à²¸à²¾à²µà²¿à²°": "thousand", "à²¸à²¾à²µà²¿à²°à²¦": "thousand"}
    CURRENCY_WORDS = {"à²°à³‚à²ªà²¾à²¯à²¿": "rupee", "à²°à³à²ªà²¾à²¯à²¿": "rupee", "à²°à³‚.": "rupee",
                      "à²°à³‚": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "à²®à³Šà²¦à²²à²¨à³‡"}
    ORDINAL_SUFFIX = "à²¨à³‡"
    ORDINAL_INPUT_SUFFIXES = ["à²¨à³†à²¯", "à²¨à³‡", "à²¨à³†"]

    # ==================================================================
    # Dravidian composition â€” fused hundreds/thousands with sandhi.
    # ==================================================================

    def hundreds(self, h, rest):
        if rest:
            return f"{self.HUNDREDS_COMB[h]} {self.two_digit(rest)}"
        return self.HUNDREDS[h]

    def under_thousand(self, n):
        if n < 100:
            return self.two_digit(n)
        return self.hundreds(n // 100, n % 100)

    def cardinal(self, n):
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n < 1000:
            return self.under_thousand(n)
        parts = []
        crore, n = divmod(n, 10**7)
        lakh, n = divmod(n, 10**5)
        thousand, n = divmod(n, 1000)
        if crore:
            # à²•à³‹à²Ÿà²¿ stays plain even when a remainder follows.
            parts.append(f"{self.cardinal(crore)} {self.CRORE}")
        if lakh:
            word = self.LAKH_COMB if (thousand or n) else self.LAKH
            parts.append(f"{self.two_digit(lakh)} {word}")
        if thousand:
            word = self.THOUSAND_COMB if n else self.THOUSAND
            if thousand == 1 and not (crore or lakh):
                # 1000 -> à²¸à²¾à²µà²¿à²°; 1947 -> à²¸à²¾à²µà²¿à²°à²¦ à²’à²‚à²¬à³ˆà²¨à³‚à²° à²¨à²²à²µà²¤à³à²¤à³‡à²³à³ (no à²’à²‚à²¦à³)
                parts.append(word)
            else:
                parts.append(f"{self.under_thousand(thousand)} {word}")
        if n:
            parts.append(self.under_thousand(n))
        return " ".join(parts)

    def year(self, y):
        """Kannada reads years as full cardinals (news/TTS style):
        1947 -> à²¸à²¾à²µà²¿à²°à²¦ à²’à²‚à²¬à³ˆà²¨à³‚à²° à²¨à²²à²µà²¤à³à²¤à³‡à²³à³, 2026 -> à²Žà²°à²¡à³ à²¸à²¾à²µà²¿à²°à²¦ à²‡à²ªà³à²ªà²¤à³à²¤à²¾à²°à³."""
        return self.cardinal(y)

    def decade(self, tens, full):
        # Genitive stem: final à³ drops before -à²° (à²¤à³Šà²‚à²¬à²¤à³à²¤à³ -> à²¤à³Šà²‚à²¬à²¤à³à²¤à²°).
        w = self.two_digit(tens)
        stem = w[:-1] if w.endswith("à³") else w
        tpl = self.DECADE_FULL if full else self.DECADE_SHORT
        return tpl.format(tens=stem)

    def _ordinal_compose(self, n, src_suffix):
        # à²Žà²°à²¡à³ -> à²Žà²°à²¡à²¨à³‡ : final à³ of the cardinal drops before à²¨à³‡.
        base = self.cardinal(n)
        if base.endswith("à³"):
            base = base[:-1]
        suffix = src_suffix if src_suffix in self.ORDINAL_INPUT_SUFFIXES else self.ORDINAL_SUFFIX
        return f"{base}{suffix}"

    # Kannada case markers attach with oblique on à³-final words:
    # à²à²¦à³+à²‡à²‚à²¦->à²à²¦à²°à²¿à²‚à²¦, à²®à³‚à²°à³+à²—à³†->à²®à³‚à²°à²•à³à²•à³†, à²’à²‚à²¬à²¤à³à²¤à³+à²¨à²²à³à²²à²¿->à²’à²‚à²¬à²¤à³à²¤à²°à²²à³à²²à²¿;
    # consonant/à³†-final just attach (à²¨à²¿à²®à²¿à²·+à²•à³à²•à³†->à²¨à²¿à²®à²¿à²·à²•à³à²•à³†, à²¸à³Šà²¨à³à²¨à³†+à²—à³†->à²¸à³Šà²¨à³à²¨à³†à²—à³†)
    _PP = {"à²—à³†", "à²•à³à²•à³†", "à²‡à²‚à²¦", "à²¦à²¿à²‚à²¦", "à²°à²¿à²‚à²¦", "à²¨à²²à³à²²à²¿", "à²…à²²à³à²²à²¿",
           "à²°à²²à³à²²à²¿", "à²µà²°à³†à²—à³†", "à²—à²¾à²—à²¿", "à²•à³à²•à²¾à²—à²¿"}

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if not last or not ("\u0c80" <= last[-1] <= "\u0cff"):
            return None
        if t == last:
            return rep
        if t not in self._PP:
            return None
        base = rep[:len(rep)-len(last)]
        dative = t in ("à²—à³†", "à²•à³à²•à³†")
        ablative = t in ("à²‡à²‚à²¦", "à²¦à²¿à²‚à²¦", "à²°à²¿à²‚à²¦")
        locative = t in ("à²¨à²²à³à²²à²¿", "à²…à²²à³à²²à²¿", "à²°à²²à³à²²à²¿")
        if last.endswith("\u0cc1"):              # à³-final -> drop, oblique -à²°/-à²•à³à²•à³†
            stem = last[:-1]
            suf = "à²•à³à²•à³†" if dative else "à²°à²¿à²‚à²¦" if ablative else "à²°à²²à³à²²à²¿" if locative else t
            return base + stem + suf
        if last.endswith(("\u0cc6", "\u0cbf", "\u0cc0")):  # à³†/à²¿/à³€-final
            suf = "à²—à³†" if dative else "à²¯à²¿à²‚à²¦" if ablative else "à²¯à²²à³à²²à²¿" if locative else t
            return base + last + suf
        # consonant/inherent-a-final -> -à²•à³à²•à³†/-à²¦à²¿à²‚à²¦/-à²¦à²²à³à²²à²¿
        suf = "à²•à³à²•à³†" if dative else "à²¦à²¿à²‚à²¦" if ablative else "à²¦à²²à³à²²à²¿" if locative else t
        return base + last + suf


class Malayalam(LangBase):
    code = "ml"
    name = "Malayalam"
    native_digits = "àµ¦àµ§àµ¨àµ©àµªàµ«àµ¬àµ­àµ®àµ¯"
    COUNT_ONE = "à´’à´°àµ"

    ONES = [
        "à´ªàµ‚à´œàµà´¯à´‚", "à´’à´¨àµà´¨àµ", "à´°à´£àµà´Ÿàµ", "à´®àµ‚à´¨àµà´¨àµ", "à´¨à´¾à´²àµ", "à´…à´žàµà´šàµ", "à´†à´±àµ", "à´à´´àµ", "à´Žà´Ÿàµà´Ÿàµ", "à´’à´®àµà´ªà´¤àµ",
        "à´ªà´¤àµà´¤àµ", "à´ªà´¤à´¿à´¨àµŠà´¨àµà´¨àµ", "à´ªà´¨àµà´¤àµà´°à´£àµà´Ÿàµ", "à´ªà´¤à´¿à´®àµ‚à´¨àµà´¨àµ", "à´ªà´¤à´¿à´¨à´¾à´²àµ", "à´ªà´¤à´¿à´¨à´žàµà´šàµ", "à´ªà´¤à´¿à´¨à´¾à´±àµ",
        "à´ªà´¤à´¿à´¨àµ‡à´´àµ", "à´ªà´¤à´¿à´¨àµ†à´Ÿàµà´Ÿàµ", "à´ªà´¤àµà´¤àµŠà´®àµà´ªà´¤àµ",
        "à´‡à´°àµà´ªà´¤àµ", "à´‡à´°àµà´ªà´¤àµà´¤à´¿à´¯àµŠà´¨àµà´¨àµ", "à´‡à´°àµà´ªà´¤àµà´¤à´¿à´°à´£àµà´Ÿàµ", "à´‡à´°àµà´ªà´¤àµà´¤à´¿à´®àµ‚à´¨àµà´¨àµ", "à´‡à´°àµà´ªà´¤àµà´¤à´¿à´¨à´¾à´²àµ",
        "à´‡à´°àµà´ªà´¤àµà´¤à´¿à´¯à´žàµà´šàµ", "à´‡à´°àµà´ªà´¤àµà´¤à´¿à´¯à´¾à´±àµ", "à´‡à´°àµà´ªà´¤àµà´¤à´¿à´¯àµ‡à´´àµ", "à´‡à´°àµà´ªà´¤àµà´¤à´¿à´¯àµ†à´Ÿàµà´Ÿàµ", "à´‡à´°àµà´ªà´¤àµà´¤à´¿à´¯àµŠà´®àµà´ªà´¤àµ",
        "à´®àµà´ªàµà´ªà´¤àµ", "à´®àµà´ªàµà´ªà´¤àµà´¤à´¿à´¯àµŠà´¨àµà´¨àµ", "à´®àµà´ªàµà´ªà´¤àµà´¤à´¿à´°à´£àµà´Ÿàµ", "à´®àµà´ªàµà´ªà´¤àµà´¤à´¿à´®àµ‚à´¨àµà´¨àµ", "à´®àµà´ªàµà´ªà´¤àµà´¤à´¿à´¨à´¾à´²àµ",
        "à´®àµà´ªàµà´ªà´¤àµà´¤à´¿à´¯à´žàµà´šàµ", "à´®àµà´ªàµà´ªà´¤àµà´¤à´¿à´¯à´¾à´±àµ", "à´®àµà´ªàµà´ªà´¤àµà´¤à´¿à´¯àµ‡à´´àµ", "à´®àµà´ªàµà´ªà´¤àµà´¤à´¿à´¯àµ†à´Ÿàµà´Ÿàµ", "à´®àµà´ªàµà´ªà´¤àµà´¤à´¿à´¯àµŠà´®àµà´ªà´¤àµ",
        "à´¨à´¾àµ½à´ªàµà´ªà´¤àµ", "à´¨à´¾àµ½à´ªàµà´ªà´¤àµà´¤à´¿à´¯àµŠà´¨àµà´¨àµ", "à´¨à´¾àµ½à´ªàµà´ªà´¤àµà´¤à´¿à´°à´£àµà´Ÿàµ", "à´¨à´¾àµ½à´ªàµà´ªà´¤àµà´¤à´¿à´®àµ‚à´¨àµà´¨àµ", "à´¨à´¾àµ½à´ªàµà´ªà´¤àµà´¤à´¿à´¨à´¾à´²àµ",
        "à´¨à´¾àµ½à´ªàµà´ªà´¤àµà´¤à´¿à´¯à´žàµà´šàµ", "à´¨à´¾àµ½à´ªàµà´ªà´¤àµà´¤à´¿à´¯à´¾à´±àµ", "à´¨à´¾àµ½à´ªàµà´ªà´¤àµà´¤à´¿à´¯àµ‡à´´àµ", "à´¨à´¾àµ½à´ªàµà´ªà´¤àµà´¤à´¿à´¯àµ†à´Ÿàµà´Ÿàµ", "à´¨à´¾àµ½à´ªàµà´ªà´¤àµà´¤à´¿à´¯àµŠà´®àµà´ªà´¤àµ",
        "à´…à´®àµà´ªà´¤àµ", "à´…à´®àµà´ªà´¤àµà´¤à´¿à´¯àµŠà´¨àµà´¨àµ", "à´…à´®àµà´ªà´¤àµà´¤à´¿à´°à´£àµà´Ÿàµ", "à´…à´®àµà´ªà´¤àµà´¤à´¿à´®àµ‚à´¨àµà´¨àµ", "à´…à´®àµà´ªà´¤àµà´¤à´¿à´¨à´¾à´²àµ",
        "à´…à´®àµà´ªà´¤àµà´¤à´¿à´¯à´žàµà´šàµ", "à´…à´®àµà´ªà´¤àµà´¤à´¿à´¯à´¾à´±àµ", "à´…à´®àµà´ªà´¤àµà´¤à´¿à´¯àµ‡à´´àµ", "à´…à´®àµà´ªà´¤àµà´¤à´¿à´¯àµ†à´Ÿàµà´Ÿàµ", "à´…à´®àµà´ªà´¤àµà´¤à´¿à´¯àµŠà´®àµà´ªà´¤àµ",
        "à´…à´±àµà´ªà´¤àµ", "à´…à´±àµà´ªà´¤àµà´¤à´¿à´¯àµŠà´¨àµà´¨àµ", "à´…à´±àµà´ªà´¤àµà´¤à´¿à´°à´£àµà´Ÿàµ", "à´…à´±àµà´ªà´¤àµà´¤à´¿à´®àµ‚à´¨àµà´¨àµ", "à´…à´±àµà´ªà´¤àµà´¤à´¿à´¨à´¾à´²àµ",
        "à´…à´±àµà´ªà´¤àµà´¤à´¿à´¯à´žàµà´šàµ", "à´…à´±àµà´ªà´¤àµà´¤à´¿à´¯à´¾à´±àµ", "à´…à´±àµà´ªà´¤àµà´¤à´¿à´¯àµ‡à´´àµ", "à´…à´±àµà´ªà´¤àµà´¤à´¿à´¯àµ†à´Ÿàµà´Ÿàµ", "à´…à´±àµà´ªà´¤àµà´¤à´¿à´¯àµŠà´®àµà´ªà´¤àµ",
        "à´Žà´´àµà´ªà´¤àµ", "à´Žà´´àµà´ªà´¤àµà´¤à´¿à´¯àµŠà´¨àµà´¨àµ", "à´Žà´´àµà´ªà´¤àµà´¤à´¿à´°à´£àµà´Ÿàµ", "à´Žà´´àµà´ªà´¤àµà´¤à´¿à´®àµ‚à´¨àµà´¨àµ", "à´Žà´´àµà´ªà´¤àµà´¤à´¿à´¨à´¾à´²àµ",
        "à´Žà´´àµà´ªà´¤àµà´¤à´¿à´¯à´žàµà´šàµ", "à´Žà´´àµà´ªà´¤àµà´¤à´¿à´¯à´¾à´±àµ", "à´Žà´´àµà´ªà´¤àµà´¤à´¿à´¯àµ‡à´´àµ", "à´Žà´´àµà´ªà´¤àµà´¤à´¿à´¯àµ†à´Ÿàµà´Ÿàµ", "à´Žà´´àµà´ªà´¤àµà´¤à´¿à´¯àµŠà´®àµà´ªà´¤àµ",
        "à´Žàµºà´ªà´¤àµ", "à´Žàµºà´ªà´¤àµà´¤à´¿à´¯àµŠà´¨àµà´¨àµ", "à´Žàµºà´ªà´¤àµà´¤à´¿à´°à´£àµà´Ÿàµ", "à´Žàµºà´ªà´¤àµà´¤à´¿à´®àµ‚à´¨àµà´¨àµ", "à´Žàµºà´ªà´¤àµà´¤à´¿à´¨à´¾à´²àµ",
        "à´Žàµºà´ªà´¤àµà´¤à´¿à´¯à´žàµà´šàµ", "à´Žàµºà´ªà´¤àµà´¤à´¿à´¯à´¾à´±àµ", "à´Žàµºà´ªà´¤àµà´¤à´¿à´¯àµ‡à´´àµ", "à´Žàµºà´ªà´¤àµà´¤à´¿à´¯àµ†à´Ÿàµà´Ÿàµ", "à´Žàµºà´ªà´¤àµà´¤à´¿à´¯àµŠà´®àµà´ªà´¤àµ",
        "à´¤àµŠà´£àµà´£àµ‚à´±àµ", "à´¤àµŠà´£àµà´£àµ‚à´±àµà´±à´¿à´¯àµŠà´¨àµà´¨àµ", "à´¤àµŠà´£àµà´£àµ‚à´±àµà´±à´¿à´°à´£àµà´Ÿàµ", "à´¤àµŠà´£àµà´£àµ‚à´±àµà´±à´¿à´®àµ‚à´¨àµà´¨àµ", "à´¤àµŠà´£àµà´£àµ‚à´±àµà´±à´¿à´¨à´¾à´²àµ",
        "à´¤àµŠà´£àµà´£àµ‚à´±àµà´±à´¿à´¯à´žàµà´šàµ", "à´¤àµŠà´£àµà´£àµ‚à´±àµà´±à´¿à´¯à´¾à´±àµ", "à´¤àµŠà´£àµà´£àµ‚à´±àµà´±à´¿à´¯àµ‡à´´àµ", "à´¤àµŠà´£àµà´£àµ‚à´±àµà´±à´¿à´¯àµ†à´Ÿàµà´Ÿàµ", "à´¤àµŠà´£àµà´£àµ‚à´±àµà´±à´¿à´¯àµŠà´®àµà´ªà´¤àµ",
    ]
    HUNDRED = "à´¨àµ‚à´±àµ"        # not used in composition (Dravidian override below)
    THOUSAND = "à´†à´¯à´¿à´°à´‚"
    LAKH = "à´²à´•àµà´·à´‚"
    CRORE = "à´•àµ‹à´Ÿà´¿"

    # Fused hundreds: standalone and combining (before a remainder).
    _H_STAND = {1: "à´¨àµ‚à´±àµ", 2: "à´‡à´°àµà´¨àµ‚à´±àµ", 3: "à´®àµà´¨àµà´¨àµ‚à´±àµ", 4: "à´¨à´¾à´¨àµ‚à´±àµ", 5: "à´…à´žàµà´žàµ‚à´±àµ",
                6: "à´…à´±àµà´¨àµ‚à´±àµ", 7: "à´Žà´´àµà´¨àµ‚à´±àµ", 8: "à´Žà´£àµà´£àµ‚à´±àµ", 9: "à´¤àµŠà´³àµà´³à´¾à´¯à´¿à´°à´‚"}
    _H_COMB = {1: "à´¨àµ‚à´±àµà´±à´¿", 2: "à´‡à´°àµà´¨àµ‚à´±àµà´±à´¿", 3: "à´®àµà´¨àµà´¨àµ‚à´±àµà´±à´¿", 4: "à´¨à´¾à´¨àµ‚à´±àµà´±à´¿", 5: "à´…à´žàµà´žàµ‚à´±àµà´±à´¿",
               6: "à´…à´±àµà´¨àµ‚à´±àµà´±à´¿", 7: "à´Žà´´àµà´¨àµ‚à´±àµà´±à´¿", 8: "à´Žà´£àµà´£àµ‚à´±àµà´±à´¿", 9: "à´¤àµŠà´³àµà´³à´¾à´¯à´¿à´°à´¤àµà´¤à´¿"}

    # Fused thousands for multipliers 1..10; others via sandhi (see _thousand).
    _TH_STAND = {1: "à´†à´¯à´¿à´°à´‚", 2: "à´°à´£àµà´Ÿà´¾à´¯à´¿à´°à´‚", 3: "à´®àµ‚à´µà´¾à´¯à´¿à´°à´‚", 4: "à´¨à´¾à´²à´¾à´¯à´¿à´°à´‚",
                 5: "à´…à´¯àµà´¯à´¾à´¯à´¿à´°à´‚", 6: "à´†à´±à´¾à´¯à´¿à´°à´‚", 7: "à´à´´à´¾à´¯à´¿à´°à´‚", 8: "à´Žà´£àµà´£à´¾à´¯à´¿à´°à´‚",
                 9: "à´’à´®àµà´ªà´¤à´¿à´¨à´¾à´¯à´¿à´°à´‚", 10: "à´ªà´¤à´¿à´¨à´¾à´¯à´¿à´°à´‚"}

    _ONE_MULT = "à´’à´°àµ"       # multiplier word for 1 lakh / 1 crore

    LETTERS = {
        "A": "à´Ž", "B": "à´¬à´¿", "C": "à´¸à´¿", "D": "à´¡à´¿", "E": "à´‡", "F": "à´Žà´«àµ",
        "G": "à´œà´¿", "H": "à´Žà´šàµà´šàµ", "I": "à´", "J": "à´œàµ†", "K": "à´•àµ†", "L": "à´Žàµ½",
        "M": "à´Žà´‚", "N": "à´Žàµ»", "O": "à´’", "P": "à´ªà´¿", "Q": "à´•àµà´¯àµ", "R": "à´†àµ¼",
        "S": "à´Žà´¸àµ", "T": "à´Ÿà´¿", "U": "à´¯àµ", "V": "à´µà´¿", "W": "à´¡à´¬àµà´²àµà´¯àµ",
        "X": "à´Žà´•àµà´¸àµ", "Y": "à´µàµˆ", "Z": "à´‡à´¸à´¡àµ",
    }

    MONTHS = ["à´œà´¨àµà´µà´°à´¿", "à´«àµ†à´¬àµà´°àµà´µà´°à´¿", "à´®à´¾àµ¼à´šàµà´šàµ", "à´à´ªàµà´°à´¿àµ½", "à´®àµ‡à´¯àµ", "à´œàµ‚àµº",
              "à´œàµ‚à´²àµˆ", "à´“à´—à´¸àµà´±àµà´±àµ", "à´¸àµ†à´ªàµà´±àµà´±à´‚à´¬àµ¼", "à´’à´•àµà´Ÿàµ‹à´¬àµ¼", "à´¨à´µà´‚à´¬àµ¼", "à´¡à´¿à´¸à´‚à´¬àµ¼"]

    SYM = {
        "rupee": "à´°àµ‚à´ª", "dollar": "à´¡àµ‹à´³àµ¼", "euro": "à´¯àµ‚à´±àµ‹", "pound": "à´ªàµ—à´£àµà´Ÿàµ",
        "cent": "à´¸àµ†à´¨àµà´±àµ",
        "percent": "à´¶à´¤à´®à´¾à´¨à´‚",
        "point": "à´¦à´¶à´¾à´‚à´¶à´‚",
        "minus": "à´®àµˆà´¨à´¸àµ", "plus": "à´ªàµà´²à´¸àµ", "equals": "à´¸à´®à´‚",
        "times": "à´—àµà´£à´‚", "divide": "à´¹à´°à´£à´‚",
        "squared": "à´µàµ¼à´—à´‚", "sqrt": "à´µàµ¼à´—à´®àµ‚à´²à´‚",
        "lt": "à´¨àµ†à´•àµà´•à´¾àµ¾ à´•àµà´±à´µàµ", "gt": "à´¨àµ†à´•àµà´•à´¾àµ¾ à´•àµ‚à´Ÿàµà´¤àµ½",
        "plus_minus": "à´ªàµà´²à´¸àµ-à´®àµˆà´¨à´¸àµ",
        "at": "à´…à´±àµà´±àµ à´¦ à´±àµ‡à´±àµà´±àµ", "underscore": "à´…à´£àµà´Ÿàµ¼à´¸àµà´•àµ‹àµ¼", "dot": "à´¡àµ‹à´Ÿàµà´Ÿàµ",
        "slash": "à´¸àµà´²à´¾à´·àµ", "hash": "à´¹à´¾à´·àµ", "amp": "à´†àµ»à´¡àµ",
        "half": "à´…à´°", "quarter": "à´•à´¾àµ½", "three_quarters": "à´®àµà´•àµà´•à´¾àµ½",
        "degree": "à´¡à´¿à´—àµà´°à´¿", "celsius": "à´¡à´¿à´—àµà´°à´¿ à´¸àµ†àµ½à´·àµà´¯à´¸àµ", "fahrenheit": "à´¡à´¿à´—àµà´°à´¿ à´«à´¾à´°àµ»à´¹àµ€à´±àµà´±àµ",
        "minute": "à´®à´¿à´¨à´¿à´±àµà´±àµ", "second": "à´¸àµ†à´•àµà´•àµ»à´¡àµ",
    }

    TIME_ON_HOUR = "{h} à´®à´£à´¿"
    TIME_HM = "{h} à´®à´£à´¿ {m} à´®à´¿à´¨à´¿à´±àµà´±àµ"
    RANGE = "{a} à´®àµà´¤àµ½ {b}"
    RANGE_TO = "{a} à´®àµà´¤àµ½ {b} à´µà´°àµ†"
    RATIO = "{a} à´…à´¨àµà´ªà´¾à´¤à´‚ {b}"
    POWER = "{base} à´˜à´¾à´¤à´‚ {exp}"
    FRACTION = "{num} à´¬àµˆ {den}"
    DECADE_SHORT = "{tens}à´•àµ¾"
    DECADE_FULL = "{tens}à´•à´³àµà´Ÿàµ† à´¦à´¶à´•à´‚"

    PERIODS = {"morning": "à´°à´¾à´µà´¿à´²àµ†", "afternoon": "à´‰à´šàµà´šà´¯àµà´•àµà´•àµ",
               "evening": "à´µàµˆà´•àµà´¨àµà´¨àµ‡à´°à´‚", "night": "à´°à´¾à´¤àµà´°à´¿"}

    TIME_MARKERS = ["à´®à´£à´¿", "à´®à´£à´¿à´•àµà´•àµ", "à´®à´£à´¿à´¯àµ‹à´Ÿàµ†"]
    MARKER_SUFFIXES = {"à´®à´£à´¿à´•àµà´•àµ": "à´¨àµ", "à´®à´£à´¿à´¯àµ‹à´Ÿàµ†": "à´“à´Ÿàµ†"}
    PERIOD_DETECT = {"à´°à´¾à´µà´¿à´²àµ†": "morning", "à´•à´¾à´²à´¤àµà´¤àµ": "morning",
                     "à´‰à´šàµà´šà´¯àµà´•àµà´•àµ": "afternoon", "à´‰à´šàµà´šà´•àµà´•àµ": "afternoon",
                     "à´‰à´šàµà´š": "afternoon", "à´‰à´šàµà´šà´•à´´à´¿à´žàµà´žàµ": "afternoon",
                     "à´µàµˆà´•àµà´¨àµà´¨àµ‡à´°à´‚": "evening", "à´µàµˆà´•à´¿à´Ÿàµà´Ÿàµ": "evening",
                     "à´¸à´¨àµà´§àµà´¯à´•àµà´•àµ": "evening", "à´°à´¾à´¤àµà´°à´¿": "night"}
    YEAR_WORDS = ["à´µàµ¼à´·à´‚", "à´µàµ¼à´·à´¤àµà´¤à´¿àµ½", "à´µàµ¼à´·à´¤àµà´¤àµ†", "à´•àµŠà´²àµà´²à´‚", "à´†à´£àµà´Ÿàµ"]
    ID_KEYWORDS = ["à´¨à´®àµà´ªàµ¼", "à´¨à´®àµà´ªà´°àµ", "à´¨à´®àµà´ªà´±à´¿àµ½", "à´…à´•àµà´•àµ—à´£àµà´Ÿàµ", "à´•àµ‹à´¡àµ", "à´“àµ¼à´¡àµ¼", "à´«àµ‹àµº", "à´®àµŠà´¬àµˆàµ½", "à´’à´Ÿà´¿à´ªà´¿", "à´ªà´¿àµ»", "à´à´¡à´¿", "à´ªà´¾à´¸àµâ€Œà´µàµ‡à´¡àµ", "à´Ÿàµà´°àµ†à´¯à´¿àµ»", "à´Žà´•àµà´¸àµà´ªàµà´°à´¸àµ", "à´ªà´¿à´Žàµ»à´†àµ¼", "à´•à´¾àµ¼à´¡àµ", "à´±à´«à´±àµ»à´¸àµ", "à´Ÿàµà´°à´¾àµ»à´¸à´¾à´•àµà´·àµ»", "à´Ÿà´¿à´•àµà´•à´±àµà´±àµ", "à´…à´•àµà´•àµ—à´£àµà´Ÿà´¿àµ½", "à´¨à´®àµà´ªà´±à´¿à´²àµ‡à´•àµà´•àµ", "à´¯àµ‚à´¸àµ¼", "à´¤à´µà´£", "à´…à´•àµà´•à´‚", "à´‡à´Ÿà´ªà´¾à´Ÿàµ"]
    NUMBER_WORD = "à´¨à´®àµà´ªàµ¼"
    ABBREV_UNITS = {"MINS":"à´®à´¿à´¨à´¿à´±àµà´±àµ", "MIN":"à´®à´¿à´¨à´¿à´±àµà´±àµ", "HRS":"à´®à´£à´¿à´•àµà´•àµ‚àµ¼", "HR":"à´®à´£à´¿à´•àµà´•àµ‚àµ¼", "SECS":"à´¸àµ†à´•àµà´•àµ»à´¡àµ", "SEC":"à´¸àµ†à´•àµà´•àµ»à´¡àµ", "KMS":"à´•à´¿à´²àµ‹à´®àµ€à´±àµà´±àµ¼"}
    LABEL_WORDS = ["à´…à´¨àµà´šàµà´›àµ‡à´¦à´‚", "à´µà´•àµà´ªàµà´ªàµ", "à´®àµà´±à´¿", "à´±àµ‚à´‚", "à´µà´¾àµ¼à´¡àµ", "à´•àµà´²à´¾à´¸àµ",
                   "à´…à´§àµà´¯à´¾à´¯à´‚", "à´­à´¾à´—à´‚", "à´—àµ‡à´±àµà´±àµ", "à´•àµ—à´£àµà´Ÿàµ¼"]
    MINUTE_WORDS = ["à´®à´¿à´¨à´¿à´±àµà´±àµ", "à´®à´¿à´¨àµà´Ÿàµà´Ÿàµ"]
    SCALE_WORDS = {"à´•àµ‹à´Ÿà´¿": "crore", "à´•àµ‹à´Ÿà´¿à´¯àµà´Ÿàµ†": "crore",
                   "à´²à´•àµà´·à´‚": "lakh", "à´²à´•àµà´·à´¤àµà´¤à´¿": "lakh", "à´²à´•àµà´·à´¤àµà´¤à´¿à´¨àµà´±àµ†": "lakh",
                   "à´†à´¯à´¿à´°à´‚": "thousand", "à´†à´¯à´¿à´°à´¤àµà´¤à´¿": "thousand"}
    CURRENCY_WORDS = {"à´°àµ‚à´ª": "rupee", "à´°àµ‚à´ªà´¾": "rupee", "à´°àµ‚à´ªà´¯àµà´Ÿàµ†": "rupee",
                      "à´°àµ‚à´ªà´¯àµà´•àµà´•àµ": "rupee"}

    ORDINAL_EXCEPTIONS = {}
    ORDINAL_SUFFIX = "à´¾à´‚"          # à´’à´¨àµà´¨àµ -> à´’à´¨àµà´¨à´¾à´‚, à´ªà´¤àµà´¤àµ -> à´ªà´¤àµà´¤à´¾à´‚
    ORDINAL_INPUT_SUFFIXES = ["à´¾à´‚", "à´¾à´®à´¤àµà´¤àµ†", "à´¾à´®à´¤àµ"]

    _VIRAMA = "àµ"   # àµ
    _ANUSVARA = "à´‚"  # à´‚

    # ------------------------------------------------------------------
    # Dravidian composition overrides (fused hundreds/thousands + sandhi)
    # ------------------------------------------------------------------

    def hundreds(self, h, rest):
        if rest:
            return f"{self._H_COMB[h]} {self.two_digit(rest)}"
        return self._H_STAND[h]

    def _thousand(self, m, combining):
        """Fused '<m> thousand' word for m in 1..99."""
        if m in self._TH_STAND:
            word = self._TH_STAND[m]
        else:
            base = self.ONES[m]
            if base.endswith("à´¤" + self._VIRAMA):
                # ends in ...à´¤àµ (round tens, -à´’à´®àµà´ªà´¤àµ): à´‡à´°àµà´ªà´¤àµ -> à´‡à´°àµà´ªà´¤à´¿à´¨à´¾à´¯à´¿à´°à´‚
                word = base[:-1] + "à´¿à´¨à´¾à´¯à´¿à´°à´‚"
            elif base.endswith(self._VIRAMA):
                # virama + à´† -> à´¾ : à´…à´®àµà´ªà´¤àµà´¤à´¿à´¨à´¾à´²àµ -> à´…à´®àµà´ªà´¤àµà´¤à´¿à´¨à´¾à´²à´¾à´¯à´¿à´°à´‚,
                # à´¤àµŠà´£àµà´£àµ‚à´±àµ -> à´¤àµŠà´£àµà´£àµ‚à´±à´¾à´¯à´¿à´°à´‚
                word = base[:-1] + "à´¾à´¯à´¿à´°à´‚"
            else:
                word = base + " " + self.THOUSAND
        if combining:
            # à´‚ -> à´¤àµà´¤à´¿ : à´†à´¯à´¿à´°à´‚ -> à´†à´¯à´¿à´°à´¤àµà´¤à´¿
            return word[:-1] + "à´¤àµà´¤à´¿"
        return word

    def cardinal(self, n):
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n < 100:
            return self.ONES[n]
        if n < 1000:
            return self.hundreds(n // 100, n % 100)
        parts = []
        crore, n = divmod(n, 10**7)
        lakh, n = divmod(n, 10**5)
        thousand, n = divmod(n, 1000)
        if crore:
            mult = self._ONE_MULT if crore == 1 else self.cardinal(crore)
            parts.append(f"{mult} {self.CRORE}")
        if lakh:
            word = self.LAKH[:-1] + "à´¤àµà´¤à´¿" if (thousand or n) else self.LAKH
            mult = self._ONE_MULT if lakh == 1 else self.two_digit(lakh)
            parts.append(f"{mult} {word}")
        if thousand:
            parts.append(self._thousand(thousand, combining=bool(n)))
        if n:
            parts.append(self.under_thousand(n))
        return " ".join(parts)

    def year(self, y):
        """Malayalam reads years as full cardinals (news/TTS convention):
        1947 -> à´†à´¯à´¿à´°à´¤àµà´¤à´¿ à´¤àµŠà´³àµà´³à´¾à´¯à´¿à´°à´¤àµà´¤à´¿ à´¨à´¾àµ½à´ªàµà´ªà´¤àµà´¤à´¿à´¯àµ‡à´´àµ."""
        return self.cardinal(y)

    def decade(self, tens, full):
        # à´¤àµŠà´£àµà´£àµ‚à´±àµ + à´•àµ¾ -> à´¤àµŠà´£àµà´£àµ‚à´±àµà´•àµ¾ (final virama -> àµ before à´•àµ¾)
        stem = self.two_digit(tens)
        if stem.endswith(self._VIRAMA):
            stem = stem[:-1] + "àµ"
        tpl = self.DECADE_FULL if full else self.DECADE_SHORT
        return tpl.format(tens=stem)

    def _ordinal_compose(self, n, src_suffix):
        # à´°à´£àµà´Ÿàµ -> à´°à´£àµà´Ÿà´¾à´‚ : strip final virama/anusvara, then attach suffix.
        stem = self.cardinal(n)
        if stem.endswith(self._VIRAMA) or stem.endswith(self._ANUSVARA):
            stem = stem[:-1]
        suffix = src_suffix if src_suffix in self.ORDINAL_INPUT_SUFFIXES else self.ORDINAL_SUFFIX
        return f"{stem}{suffix}"

    # Malayalam case markers (often hyphen-attached in source) attach with
    # sandhi: à´°àµ‚à´ª+à´¨àµà´±àµ†->à´°àµ‚à´ªà´¯àµà´Ÿàµ†, à´†à´±àµ+àµ½->à´†à´±à´¿àµ½, à´°àµ‚à´ª+àµ½->à´°àµ‚à´ªà´¯à´¿àµ½
    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if not last or not ("à´€" <= last[-1] <= "àµ¿"):
            return None
        base = rep[:len(rep)-len(last)]
        virama = last.endswith("àµ")
        anusvara = last.endswith("à´‚")
        ivow = last[-1] in ("à´¿", "àµ€")
        if t in ("à´¨àµà´±àµ†", "àµà´Ÿàµ†", "à´¯àµà´Ÿàµ†"):
            if virama:   return base + last[:-1] + "à´¿à´¨àµà´±àµ†"
            if anusvara: return base + last[:-1] + "à´¤àµà´¤à´¿à´¨àµà´±àµ†"
            return base + last + "à´¯àµà´Ÿàµ†"
        if t in ("àµ½", "à´¿àµ½", "à´²àµ†", "à´¿à´²àµ†"):
            end = "à´¿à´²àµ†" if t in ("à´²àµ†", "à´¿à´²àµ†") else "à´¿àµ½"
            if virama:   return base + last[:-1] + end
            if anusvara: return base + last[:-1] + "à´¤àµà´¤" + end
            return base + last + "à´¯" + end
        if t in ("à´²àµ‡à´•àµà´•àµ", "à´¿à´²àµ‡à´•àµà´•àµ"):
            if virama:   return base + last[:-1] + "à´¿à´²àµ‡à´•àµà´•àµ"
            if anusvara: return base + last[:-1] + "à´¤àµà´¤à´¿à´²àµ‡à´•àµà´•àµ"
            return base + last + "à´¯à´¿à´²àµ‡à´•àµà´•àµ"
        if t in ("à´¨àµ", "à´¿à´¨àµ", "à´•àµà´•àµ", "à´¿à´¨à´¾à´£àµ"):
            tail = "à´¿à´¨à´¾à´£àµ" if t == "à´¿à´¨à´¾à´£àµ" else "à´¿à´¨àµ"
            if virama:   return base + last[:-1] + tail
            if anusvara: return base + last[:-1] + "à´¤àµà´¤" + tail
            if ivow:     return base + last + "à´•àµà´•àµ"
            return base + last + "à´¯àµà´•àµà´•àµ"
        return None


class Marathi(LangBase):
    code = "mr"
    name = "Marathi"
    native_digits = "à¥¦à¥§à¥¨à¥©à¥ªà¥«à¥¬à¥­à¥®à¥¯"
    SINGULAR_NOUNS = {"à¤°à¥à¤ªà¤¯à¥‡": "à¤°à¥à¤ªà¤¯à¤¾", "à¤®à¤¿à¤¨à¤¿à¤Ÿà¥‡": "à¤®à¤¿à¤¨à¤¿à¤Ÿ"}

    ONES = [
        "à¤¶à¥‚à¤¨à¥à¤¯", "à¤à¤•", "à¤¦à¥‹à¤¨", "à¤¤à¥€à¤¨", "à¤šà¤¾à¤°", "à¤ªà¤¾à¤š", "à¤¸à¤¹à¤¾", "à¤¸à¤¾à¤¤", "à¤†à¤ ", "à¤¨à¤Š",
        "à¤¦à¤¹à¤¾", "à¤…à¤•à¤°à¤¾", "à¤¬à¤¾à¤°à¤¾", "à¤¤à¥‡à¤°à¤¾", "à¤šà¥Œà¤¦à¤¾", "à¤ªà¤‚à¤§à¤°à¤¾", "à¤¸à¥‹à¤³à¤¾", "à¤¸à¤¤à¤°à¤¾", "à¤…à¤ à¤°à¤¾", "à¤à¤•à¥‹à¤£à¥€à¤¸",
        "à¤µà¥€à¤¸", "à¤à¤•à¤µà¥€à¤¸", "à¤¬à¤¾à¤µà¥€à¤¸", "à¤¤à¥‡à¤µà¥€à¤¸", "à¤šà¥‹à¤µà¥€à¤¸", "à¤ªà¤‚à¤šà¤µà¥€à¤¸", "à¤¸à¤µà¥à¤µà¥€à¤¸", "à¤¸à¤¤à¥à¤¤à¤¾à¤µà¥€à¤¸", "à¤…à¤ à¥à¤ à¤¾à¤µà¥€à¤¸", "à¤à¤•à¥‹à¤£à¤¤à¥€à¤¸",
        "à¤¤à¥€à¤¸", "à¤à¤•à¤¤à¥€à¤¸", "à¤¬à¤¤à¥à¤¤à¥€à¤¸", "à¤¤à¥‡à¤¹à¤¤à¥€à¤¸", "à¤šà¥Œà¤¤à¥€à¤¸", "à¤ªà¤¸à¥à¤¤à¥€à¤¸", "à¤›à¤¤à¥à¤¤à¥€à¤¸", "à¤¸à¤¦à¤¤à¥€à¤¸", "à¤…à¤¡à¤¤à¥€à¤¸", "à¤à¤•à¥‹à¤£à¤šà¤¾à¤³à¥€à¤¸",
        "à¤šà¤¾à¤³à¥€à¤¸", "à¤à¤•à¥à¤•à¥‡à¤šà¤¾à¤³à¥€à¤¸", "à¤¬à¥‡à¤šà¤¾à¤³à¥€à¤¸", "à¤¤à¥à¤°à¥‡à¤šà¤¾à¤³à¥€à¤¸", "à¤šà¤µà¥à¤µà¥‡à¤šà¤¾à¤³à¥€à¤¸", "à¤ªà¤‚à¤šà¥‡à¤šà¤¾à¤³à¥€à¤¸", "à¤¸à¥‡à¤¹à¥‡à¤šà¤¾à¤³à¥€à¤¸", "à¤¸à¤¤à¥à¤¤à¥‡à¤šà¤¾à¤³à¥€à¤¸", "à¤…à¤ à¥à¤ à¥‡à¤šà¤¾à¤³à¥€à¤¸", "à¤à¤•à¥‹à¤£à¤ªà¤¨à¥à¤¨à¤¾à¤¸",
        "à¤ªà¤¨à¥à¤¨à¤¾à¤¸", "à¤à¤•à¥à¤•à¤¾à¤µà¤¨à¥à¤¨", "à¤¬à¤¾à¤µà¤¨à¥à¤¨", "à¤¤à¥à¤°à¥‡à¤ªà¤¨à¥à¤¨", "à¤šà¥‹à¤ªà¤¨à¥à¤¨", "à¤ªà¤‚à¤šà¤¾à¤µà¤¨à¥à¤¨", "à¤›à¤ªà¥à¤ªà¤¨à¥à¤¨", "à¤¸à¤¤à¥à¤¤à¤¾à¤µà¤¨à¥à¤¨", "à¤…à¤ à¥à¤ à¤¾à¤µà¤¨à¥à¤¨", "à¤à¤•à¥‹à¤£à¤¸à¤¾à¤ ",
        "à¤¸à¤¾à¤ ", "à¤à¤•à¤¸à¤·à¥à¤Ÿ", "à¤¬à¤¾à¤¸à¤·à¥à¤Ÿ", "à¤¤à¥à¤°à¥‡à¤¸à¤·à¥à¤Ÿ", "à¤šà¥Œà¤¸à¤·à¥à¤Ÿ", "à¤ªà¤¾à¤¸à¤·à¥à¤Ÿ", "à¤¸à¤¹à¤¾à¤¸à¤·à¥à¤Ÿ", "à¤¸à¤¦à¥à¤¸à¤·à¥à¤Ÿ", "à¤…à¤¡à¥à¤¸à¤·à¥à¤Ÿ", "à¤à¤•à¥‹à¤£à¤¸à¤¤à¥à¤¤à¤°",
        "à¤¸à¤¤à¥à¤¤à¤°", "à¤à¤•à¥à¤•à¤¾à¤¹à¤¤à¥à¤¤à¤°", "à¤¬à¤¾à¤¹à¤¤à¥à¤¤à¤°", "à¤¤à¥à¤°à¥à¤¯à¤¾à¤¹à¤¤à¥à¤¤à¤°", "à¤šà¥Œà¤±à¥à¤¯à¤¾à¤¹à¤¤à¥à¤¤à¤°", "à¤ªà¤‚à¤šà¥à¤¯à¤¾à¤¹à¤¤à¥à¤¤à¤°", "à¤¶à¤¹à¤¾à¤¤à¥à¤¤à¤°", "à¤¸à¤¤à¥à¤¤à¥à¤¯à¤¾à¤¹à¤¤à¥à¤¤à¤°", "à¤…à¤ à¥à¤ à¥à¤¯à¤¾à¤¹à¤¤à¥à¤¤à¤°", "à¤à¤•à¥‹à¤£à¤à¤‚à¤¶à¥€",
        "à¤à¤‚à¤¶à¥€", "à¤à¤•à¥à¤•à¥à¤¯à¤¾à¤à¤‚à¤¶à¥€", "à¤¬à¥à¤¯à¤¾à¤à¤‚à¤¶à¥€", "à¤¤à¥à¤°à¥à¤¯à¤¾à¤à¤‚à¤¶à¥€", "à¤šà¥Œà¤±à¥à¤¯à¤¾à¤à¤‚à¤¶à¥€", "à¤ªà¤‚à¤šà¥à¤¯à¤¾à¤à¤‚à¤¶à¥€", "à¤¶à¤¹à¤¾à¤à¤‚à¤¶à¥€", "à¤¸à¤¤à¥à¤¤à¥à¤¯à¤¾à¤à¤‚à¤¶à¥€", "à¤…à¤ à¥à¤ à¥à¤¯à¤¾à¤à¤‚à¤¶à¥€", "à¤à¤•à¥‹à¤£à¤¨à¤µà¥à¤µà¤¦",
        "à¤¨à¤µà¥à¤µà¤¦", "à¤à¤•à¥à¤•à¥à¤¯à¤¾à¤£à¥à¤£à¤µ", "à¤¬à¥à¤¯à¤¾à¤£à¥à¤£à¤µ", "à¤¤à¥à¤°à¥à¤¯à¤¾à¤£à¥à¤£à¤µ", "à¤šà¥Œà¤±à¥à¤¯à¤¾à¤£à¥à¤£à¤µ", "à¤ªà¤‚à¤šà¥à¤¯à¤¾à¤£à¥à¤£à¤µ", "à¤¶à¤¹à¤¾à¤£à¥à¤£à¤µ", "à¤¸à¤¤à¥à¤¤à¥à¤¯à¤¾à¤£à¥à¤£à¤µ", "à¤…à¤ à¥à¤ à¥à¤¯à¤¾à¤£à¥à¤£à¤µ", "à¤¨à¤µà¥à¤µà¥à¤¯à¤¾à¤£à¥à¤£à¤µ",
    ]
    HUNDRED = "à¤¶à¥‡"
    THOUSAND = "à¤¹à¤œà¤¾à¤°"
    LAKH = "à¤²à¤¾à¤–"
    CRORE = "à¤•à¥‹à¤Ÿà¥€"

    LETTERS = {
        "A": "à¤", "B": "à¤¬à¥€", "C": "à¤¸à¥€", "D": "à¤¡à¥€", "E": "à¤ˆ", "F": "à¤à¤«",
        "G": "à¤œà¥€", "H": "à¤à¤š", "I": "à¤†à¤¯", "J": "à¤œà¥‡", "K": "à¤•à¥‡", "L": "à¤à¤²",
        "M": "à¤à¤®", "N": "à¤à¤¨", "O": "à¤“", "P": "à¤ªà¥€", "Q": "à¤•à¥à¤¯à¥‚", "R": "à¤†à¤°",
        "S": "à¤à¤¸", "T": "à¤Ÿà¥€", "U": "à¤¯à¥‚", "V": "à¤µà¥à¤¹à¥€", "W": "à¤¡à¤¬à¥à¤²à¥à¤¯à¥‚",
        "X": "à¤à¤•à¥à¤¸", "Y": "à¤µà¤¾à¤¯", "Z": "à¤à¥‡à¤¡",
    }

    MONTHS = ["à¤œà¤¾à¤¨à¥‡à¤µà¤¾à¤°à¥€", "à¤«à¥‡à¤¬à¥à¤°à¥à¤µà¤¾à¤°à¥€", "à¤®à¤¾à¤°à¥à¤š", "à¤à¤ªà¥à¤°à¤¿à¤²", "à¤®à¥‡", "à¤œà¥‚à¤¨",
              "à¤œà¥à¤²à¥ˆ", "à¤‘à¤—à¤¸à¥à¤Ÿ", "à¤¸à¤ªà¥à¤Ÿà¥‡à¤‚à¤¬à¤°", "à¤‘à¤•à¥à¤Ÿà¥‹à¤¬à¤°", "à¤¨à¥‹à¤µà¥à¤¹à¥‡à¤‚à¤¬à¤°", "à¤¡à¤¿à¤¸à¥‡à¤‚à¤¬à¤°"]

    SYM = {
        "rupee": "à¤°à¥à¤ªà¤¯à¥‡", "dollar": "à¤¡à¥‰à¤²à¤°", "euro": "à¤¯à¥à¤°à¥‹", "pound": "à¤ªà¤¾à¤‰à¤‚à¤¡",
        "cent": "à¤¸à¥‡à¤‚à¤Ÿ",
        "percent": "à¤Ÿà¤•à¥à¤•à¥‡",
        "point": "à¤ªà¥‚à¤°à¥à¤£à¤¾à¤‚à¤•",
        "minus": "à¤‰à¤£à¥‡", "plus": "à¤…à¤§à¤¿à¤•", "equals": "à¤¬à¤°à¥‹à¤¬à¤°",
        "times": "à¤—à¥à¤£à¤¿à¤²à¥‡", "divide": "à¤­à¤¾à¤—à¤¿à¤²à¥‡",
        "squared": "à¤šà¤¾ à¤µà¤°à¥à¤—", "sqrt": "à¤µà¤°à¥à¤—à¤®à¥‚à¤³",
        "lt": "à¤ªà¥‡à¤•à¥à¤·à¤¾ à¤•à¤®à¥€", "gt": "à¤ªà¥‡à¤•à¥à¤·à¤¾ à¤œà¤¾à¤¸à¥à¤¤",
        "plus_minus": "à¤…à¤§à¤¿à¤•-à¤‰à¤£à¥‡",
        "at": "à¥²à¤Ÿ à¤¦ à¤°à¥‡à¤Ÿ", "underscore": "à¤…à¤‚à¤¡à¤°à¤¸à¥à¤•à¥‹à¤°", "dot": "à¤¡à¥‰à¤Ÿ",
        "slash": "à¤¸à¥à¤²à¥…à¤¶", "hash": "à¤¹à¥…à¤¶", "amp": "à¤†à¤£à¤¿",
        "half": "à¤…à¤°à¥à¤§à¤¾", "quarter": "à¤ªà¤¾à¤µ", "three_quarters": "à¤ªà¤¾à¤Šà¤£",
        "degree": "à¤…à¤‚à¤¶", "celsius": "à¤…à¤‚à¤¶ à¤¸à¥‡à¤²à¥à¤¸à¤¿à¤…à¤¸", "fahrenheit": "à¤…à¤‚à¤¶ à¤«à¥…à¤°à¤¨à¤¹à¤¾à¤‡à¤Ÿ",
        "minute": "à¤®à¤¿à¤¨à¤¿à¤Ÿà¥‡", "second": "à¤¸à¥‡à¤•à¤‚à¤¦",
    }

    TIME_ON_HOUR = "{h} à¤µà¤¾à¤œà¤¤à¤¾"
    TIME_HM = "{h} à¤µà¤¾à¤œà¥‚à¤¨ {m} à¤®à¤¿à¤¨à¤¿à¤Ÿà¥‡"
    RANGE = "{a} à¤¤à¥‡ {b}"
    RANGE_TO = "{a} à¤¤à¥‡ {b} à¤ªà¤°à¥à¤¯à¤‚à¤¤"
    RATIO = "{a} à¤—à¥à¤£à¥‹à¤¤à¥à¤¤à¤° {b}"
    POWER = "{base} à¤šà¤¾ à¤˜à¤¾à¤¤ {exp}"
    FRACTION = "{num} à¤¬à¤Ÿà¤¾ {den}"
    DECADE_SHORT = "{tens} à¤šà¥à¤¯à¤¾"
    DECADE_FULL = "{tens} à¤šà¥‡ à¤¦à¤¶à¤•"

    PERIODS = {"morning": "à¤¸à¤•à¤¾à¤³à¥€", "afternoon": "à¤¦à¥à¤ªà¤¾à¤°à¥€",
               "evening": "à¤¸à¤‚à¤§à¥à¤¯à¤¾à¤•à¤¾à¤³à¥€", "night": "à¤°à¤¾à¤¤à¥à¤°à¥€"}

    TIME_MARKERS = ["à¤µà¤¾à¤œà¤¤à¤¾", "à¤µà¤¾à¤œà¥‚à¤¨", "à¤µà¤¾à¤œà¤²à¥‡", "à¤µà¤¾à¤œà¤²à¥à¤¯à¤¾", "à¤µà¤¾à¤œà¥‡à¤ªà¤°à¥à¤¯à¤‚à¤¤"]
    PERIOD_DETECT = {"à¤¸à¤•à¤¾à¤³à¥€": "morning", "à¤¸à¤•à¤¾à¤³": "morning", "à¤¸à¤•à¤¾à¤³à¤šà¥‡": "morning",
                     "à¤ªà¤¹à¤¾à¤Ÿà¥‡": "morning",
                     "à¤¦à¥à¤ªà¤¾à¤°à¥€": "afternoon", "à¤¦à¥à¤ªà¤¾à¤°": "afternoon", "à¤¦à¥à¤ªà¤¾à¤°à¤šà¥‡": "afternoon",
                     "à¤¸à¤‚à¤§à¥à¤¯à¤¾à¤•à¤¾à¤³à¥€": "evening", "à¤¸à¤‚à¤§à¥à¤¯à¤¾à¤•à¤¾à¤³": "evening",
                     "à¤¸à¤¾à¤¯à¤‚à¤•à¤¾à¤³à¥€": "evening",
                     "à¤°à¤¾à¤¤à¥à¤°à¥€": "night", "à¤°à¤¾à¤¤à¥à¤°": "night", "à¤°à¤¾à¤¤à¥à¤°à¥€à¤šà¥‡": "night"}
    YEAR_WORDS = ["à¤¸à¤¾à¤²", "à¤¸à¤¾à¤²à¥€", "à¤¸à¤¾à¤²à¤šà¥à¤¯à¤¾", "à¤¸à¤¨", "à¤µà¤°à¥à¤·", "à¤µà¤°à¥à¤·à¥€", "à¤‡à¤¸à¤µà¥€"]
    ID_KEYWORDS = ["à¤¨à¤‚à¤¬à¤°", "à¤•à¥à¤°à¤®à¤¾à¤‚à¤•", "à¤–à¤¾à¤¤à¥‡", "à¤–à¤¾à¤¤à¥à¤¯à¤¾à¤šà¤¾", "à¤–à¤¾à¤¤à¥à¤¯à¤¾à¤µà¤°", "à¤•à¥‹à¤¡", "à¤‘à¤°à¥à¤¡à¤°", "à¤«à¥‹à¤¨", "à¤®à¥‹à¤¬à¤¾à¤ˆà¤²", "à¤®à¥‹à¤¬à¤¾à¤‡à¤²", "à¤…à¤•à¤¾à¤‰à¤‚à¤Ÿ", "à¤…à¤•à¤¾à¤‰à¤‚à¤Ÿà¤¸à¤¾à¤ à¥€", "à¤…à¤•à¤¾à¤‰à¤‚à¤Ÿà¤®à¤§à¥‚à¤¨", "à¤–à¤¾à¤¤à¥à¤¯à¤¾à¤¤à¥‚à¤¨", "à¤–à¤¾à¤¤à¥à¤¯à¤¾à¤¤", "à¤–à¤¾à¤¤à¥à¤¯à¤¾à¤šà¥€", "à¤ªà¤¿à¤¨", "à¤†à¤¯à¤¡à¥€", "à¤ªà¤¾à¤¸à¤µà¤°à¥à¤¡", "à¤Ÿà¥à¤°à¥‡à¤¨", "à¤à¤•à¥à¤¸à¤ªà¥à¤°à¥‡à¤¸", "à¤ªà¥€à¤à¤¨à¤†à¤°", "à¤•à¤¾à¤°à¥à¤¡", "à¤°à¥‡à¤«à¤°à¤¨à¥à¤¸", "à¤Ÿà¥à¤°à¤¾à¤‚à¤à¥…à¤•à¥à¤¶à¤¨", "à¤¤à¤¿à¤•à¥€à¤Ÿ", "à¤“à¤Ÿà¥€à¤ªà¥€", "à¤—à¤¾à¤¡à¥€", "à¤¯à¥à¤œà¤°", "à¤¯à¥‚à¤œà¤°", "à¤¹à¤ªà¥à¤¤à¤¾", "à¤¹à¤ªà¥à¤¤à¥à¤¯à¤¾à¤šà¤¾", "à¤…à¤‚à¤•", "à¤µà¥à¤¯à¤µà¤¹à¤¾à¤°", "à¤Ÿà¥à¤°à¤¾à¤¨à¥à¤à¥…à¤•à¥à¤¶à¤¨", "à¤•à¥à¤°", "à¤•à¥à¤°à¤‚", "à¤šà¥‡à¤•", "à¤§à¤¨à¤¾à¤¦à¥‡à¤¶"]
    ID_AFTER_WORDS = ["à¤¡à¤¾à¤¯à¤²", "à¤•à¥‰à¤²", "à¤¸à¤‚à¤ªà¤°à¥à¤•"]
    NUMBER_WORD = "à¤•à¥à¤°à¤®à¤¾à¤‚à¤•"
    ABBREV_UNITS = {"MINS":"à¤®à¤¿à¤¨à¤¿à¤Ÿà¥‡", "MIN":"à¤®à¤¿à¤¨à¤¿à¤Ÿ", "HRS":"à¤¤à¤¾à¤¸", "HR":"à¤¤à¤¾à¤¸", "SECS":"à¤¸à¥‡à¤•à¤‚à¤¦", "SEC":"à¤¸à¥‡à¤•à¤‚à¤¦", "KMS":"à¤•à¤¿à¤²à¥‹à¤®à¥€à¤Ÿà¤°"}
    LABEL_WORDS = ["à¤•à¤²à¤®", "à¤…à¤¨à¥à¤šà¥à¤›à¥‡à¤¦", "à¤–à¥‹à¤²à¥€", "à¤µà¤°à¥à¤—", "à¤‡à¤¯à¤¤à¥à¤¤à¤¾", "à¤µà¥‰à¤°à¥à¤¡",
                   "à¤…à¤§à¥à¤¯à¤¾à¤¯", "à¤­à¤¾à¤—", "à¤ªà¥à¤°à¤•à¤°à¤£", "à¤—à¤Ÿ"]
    MINUTE_WORDS = ["à¤®à¤¿à¤¨à¤¿à¤Ÿà¥‡", "à¤®à¤¿à¤¨à¤¿à¤Ÿà¤‚", "à¤®à¤¿à¤¨à¤¿à¤Ÿ", "à¤®à¤¿à¤¨à¤¿à¤Ÿà¤¾à¤‚à¤¨à¥€"]
    SCALE_WORDS = {"à¤•à¥‹à¤Ÿà¥€": "crore", "à¤²à¤¾à¤–": "lakh", "à¤¹à¤œà¤¾à¤°": "thousand",
                   "à¤…à¤¬à¥à¤œ": "arab"}
    CURRENCY_WORDS = {"à¤°à¥à¤ªà¤¯à¥‡": "rupee", "à¤°à¥à¤ªà¤¯à¤¾": "rupee", "à¤°à¥à¤ªà¤¯à¤¾à¤‚à¤šà¤¾": "rupee",
                      "à¤°à¥à¤ªà¤¯à¤¾à¤‚à¤šà¥€": "rupee", "à¤°à¥": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "à¤ªà¤¹à¤¿à¤²à¤¾", 2: "à¤¦à¥à¤¸à¤°à¤¾", 3: "à¤¤à¤¿à¤¸à¤°à¤¾", 4: "à¤šà¥Œà¤¥à¤¾"}
    ORDINAL_SUFFIX = "à¤µà¤¾"
    ORDINAL_INPUT_SUFFIXES = ["à¤µà¤¾", "à¤µà¥€", "à¤µà¥‡", "à¤µà¥à¤¯à¤¾", "à¤µà¤‚"]

    # ------------------------------------------------------------------
    # Hundreds fuse: à¤à¤•+à¤¶à¥‡ -> à¤à¤•à¤¶à¥‡, à¤¦à¥‹à¤¨+à¤¶à¥‡ -> à¤¦à¥‹à¤¨à¤¶à¥‡ (written solid).
    # Exactly 100 with no remainder is à¤¶à¤‚à¤­à¤°.
    # ------------------------------------------------------------------
    def hundreds(self, h, rest):
        if h == 1 and rest == 0:
            return "à¤¶à¤‚à¤­à¤°"
        out = f"{self.ONES[h]}{self.HUNDRED}"
        if rest:
            out += f" {self.two_digit(rest)}"
        return out

    def year(self, y):
        """1947 -> à¤à¤•à¥‹à¤£à¥€à¤¸à¤¶à¥‡ à¤¸à¤¤à¥à¤¤à¥‡à¤šà¤¾à¤³à¥€à¤¸ (fused à¤¶à¥‡); 2026 -> à¤¦à¥‹à¤¨ à¤¹à¤œà¤¾à¤° à¤¸à¤µà¥à¤µà¥€à¤¸."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            out = f"{self.two_digit(hi)}{self.HUNDRED}"
            if lo:
                out += f" {self.two_digit(lo)}"
            return out
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # 12à¤µà¤¾ -> à¤¬à¤¾à¤°à¤¾à¤µà¤¾, 12à¤µà¥€ -> à¤¬à¤¾à¤°à¤¾à¤µà¥€ : cardinal + preserved suffix
        if src_suffix:
            return f"{self.cardinal(n)}{src_suffix}"
        if n in self.ORDINAL_EXCEPTIONS:
            return self.ORDINAL_EXCEPTIONS[n]
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"

    # Marathi: -à¤ ending noun-words take oblique plural -à¤¯à¤¾à¤‚ before a case
    # marker (à¤°à¥à¤ªà¤¯à¥‡+à¤šà¤¾->à¤°à¥à¤ªà¤¯à¤¾à¤‚à¤šà¤¾, à¤®à¤¿à¤¨à¤¿à¤Ÿà¥‡+à¤²à¤¾->à¤®à¤¿à¤¨à¤¿à¤Ÿà¤¾à¤‚à¤¨à¤¾)
    _ATTACH_NOUNS = {"à¤°à¥à¤ªà¤¯à¥‡", "à¤ªà¥ˆà¤¸à¥‡", "à¤¡à¥‰à¤²à¤°", "à¤Ÿà¤•à¥à¤•à¥‡", "à¤®à¤¿à¤¨à¤¿à¤Ÿà¥‡", "à¤¤à¤¾à¤¸",
                     "à¤•à¤¿à¤²à¥‹à¤®à¥€à¤Ÿà¤°", "à¤•à¤¿à¤²à¥‹", "à¤®à¥€à¤Ÿà¤°", "à¤²à¤¿à¤Ÿà¤°", "à¤—à¥à¤°à¥…à¤®", "à¤…à¤‚à¤¶"}
    _CASE = {"à¤šà¤¾":"à¤šà¤¾","à¤šà¥€":"à¤šà¥€","à¤šà¥‡":"à¤šà¥‡","à¤šà¤‚":"à¤šà¤‚","à¤šà¥à¤¯à¤¾":"à¤šà¥à¤¯à¤¾",
             "à¤²à¤¾":"à¤¨à¤¾","à¤¨à¤¾":"à¤¨à¤¾","à¤¨à¥€":"à¤¨à¥€","à¤¤":"à¤¤","à¤¤à¥‚à¤¨":"à¤¤à¥‚à¤¨","à¤¹à¥‚à¤¨":"à¤¹à¥‚à¤¨",
             "à¤ªà¤°à¥à¤¯à¤‚à¤¤":"à¤ªà¤°à¥à¤¯à¤‚à¤¤","à¤®à¤§à¥à¤¯à¥‡":"à¤®à¤§à¥à¤¯à¥‡","à¤¸à¤¾à¤ à¥€":"à¤¸à¤¾à¤ à¥€"}

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-").lstrip("à¤¾à¤‚à¤‚")
        last = rep.rsplit(" ", 1)[-1]
        if last not in self._ATTACH_NOUNS or t not in self._CASE:
            return None
        base = rep[:len(rep)-len(last)]
        obl = last[:-1] + "à¤¾à¤‚" if last.endswith("à¥‡") else last
        return base + obl + self._CASE[t]


class Odia(LangBase):
    code = "or"
    name = "Odia"
    native_digits = "à­¦à­§à­¨à­©à­ªà­«à­¬à­­à­®à­¯"

    ONES = [
        "à¬¶à­‚à¬¨à­à­Ÿ", "à¬à¬•", "à¬¦à­à¬‡", "à¬¤à¬¿à¬¨à¬¿", "à¬šà¬¾à¬°à¬¿", "à¬ªà¬¾à¬žà­à¬š", "à¬›à¬…", "à¬¸à¬¾à¬¤", "à¬†à¬ ", "à¬¨à¬…",
        "à¬¦à¬¶", "à¬à¬—à¬¾à¬°", "à¬¬à¬¾à¬°", "à¬¤à­‡à¬°", "à¬šà¬‰à¬¦", "à¬ªà¬¨à­à¬¦à¬°", "à¬·à­‹à¬¹à¬³", "à¬¸à¬¤à¬°", "à¬…à¬ à¬°", "à¬‰à¬£à­‡à¬‡à¬¶à¬¿",
        "à¬•à­‹à¬¡à¬¼à¬¿à¬", "à¬à¬•à­‹à¬‡à¬¶à¬¿", "à¬¬à¬¾à¬‡à¬¶à¬¿", "à¬¤à­‡à¬‡à¬¶à¬¿", "à¬šà¬¬à¬¿à¬¶à¬¿", "à¬ªà¬šà¬¿à¬¶à¬¿", "à¬›à¬¬à¬¿à¬¶à¬¿", "à¬¸à¬¤à¬¾à¬‡à¬¶à¬¿", "à¬…à¬ à­‡à¬‡à¬¶à¬¿", "à¬…à¬£à¬¤à¬¿à¬°à¬¿à¬¶à¬¿",
        "à¬¤à¬¿à¬°à¬¿à¬¶à¬¿", "à¬à¬•à¬¤à¬¿à¬°à¬¿à¬¶à¬¿", "à¬¬à¬¤à¬¿à¬¶à¬¿", "à¬¤à­‡à¬¤à¬¿à¬¶à¬¿", "à¬šà¬‰à¬¤à¬¿à¬°à¬¿à¬¶à¬¿", "à¬ªà¬žà­à¬šà¬¤à¬¿à¬°à¬¿à¬¶à¬¿", "à¬›à¬¤à¬¿à¬¶à¬¿", "à¬¸à¬‡à¬à¬¤à¬¿à¬°à¬¿à¬¶à¬¿", "à¬…à¬ à¬¤à¬¿à¬°à¬¿à¬¶à¬¿", "à¬…à¬£à¬šà¬¾à¬³à¬¿à¬¶à¬¿",
        "à¬šà¬¾à¬³à¬¿à¬¶à¬¿", "à¬à¬•à¬šà¬¾à¬³à¬¿à¬¶à¬¿", "à¬¬à­Ÿà¬¾à¬³à¬¿à¬¶à¬¿", "à¬¤à­‡à­Ÿà¬¾à¬³à¬¿à¬¶à¬¿", "à¬šà¬‰à¬°à¬¾à¬³à¬¿à¬¶à¬¿", "à¬ªà¬žà­à¬šà¬šà¬¾à¬³à¬¿à¬¶à¬¿", "à¬›à­Ÿà¬¾à¬³à¬¿à¬¶à¬¿", "à¬¸à¬¤à¬šà¬¾à¬³à¬¿à¬¶à¬¿", "à¬…à¬ à¬šà¬¾à¬³à¬¿à¬¶à¬¿", "à¬…à¬£à¬šà¬¾à¬¶",
        "à¬ªà¬šà¬¾à¬¶", "à¬à¬•à¬¾à¬¬à¬¨", "à¬¬à¬¾à¬‰à¬¨", "à¬¤à­‡à¬ªà¬¨", "à¬šà¬‰à¬¬à¬¨", "à¬ªà¬žà­à¬šà¬¾à¬¬à¬¨", "à¬›à¬ªà¬¨", "à¬¸à¬¤à¬¾à¬¬à¬¨", "à¬…à¬ à¬¾à¬¬à¬¨", "à¬…à¬£à¬·à¬ à¬¿",
        "à¬·à¬¾à¬ à¬¿à¬", "à¬à¬•à¬·à¬ à¬¿", "à¬¬à¬¾à¬·à¬ à¬¿", "à¬¤à­‡à¬·à¬ à¬¿", "à¬šà¬‰à¬·à¬ à¬¿", "à¬ªà¬žà­à¬šà¬·à¬ à¬¿", "à¬›à¬…à¬·à¬ à¬¿", "à¬¸à¬¤à¬·à¬ à¬¿", "à¬…à¬ à¬·à¬ à¬¿", "à¬…à¬£à¬¸à­à¬¤à¬°à¬¿",
        "à¬¸à¬¤à­à¬°à¬¿", "à¬à¬•à¬¸à­à¬¤à¬°à¬¿", "à¬¬à¬¾à¬¸à­à¬¤à¬°à¬¿", "à¬¤à­‡à¬¸à­à¬¤à¬°à¬¿", "à¬šà¬‰à¬¸à­à¬¤à¬°à¬¿", "à¬ªà¬žà­à¬šà¬¸à­à¬¤à¬°à¬¿", "à¬›à¬…à¬¸à­à¬¤à¬°à¬¿", "à¬¸à¬¤à¬¸à­à¬¤à¬°à¬¿", "à¬…à¬ à¬¸à­à¬¤à¬°à¬¿", "à¬…à¬£à¬¾à¬…à¬¶à­€",
        "à¬…à¬¶à­€", "à¬à¬•à¬¾à¬…à¬¶à­€", "à¬¬à­Ÿà¬¾à¬…à¬¶à­€", "à¬¤à­‡à­Ÿà¬¾à¬…à¬¶à­€", "à¬šà¬‰à¬°à¬¾à¬…à¬¶à­€", "à¬ªà¬žà­à¬šà¬¾à¬…à¬¶à­€", "à¬›à­Ÿà¬¾à¬…à¬¶à­€", "à¬¸à¬¤à¬¾à¬…à¬¶à­€", "à¬…à¬ à¬¾à¬…à¬¶à­€", "à¬…à¬£à¬¾à¬¨à¬¬à­‡",
        "à¬¨à¬¬à­‡", "à¬à¬•à¬¾à¬¨à¬¬à­‡", "à¬¬à­Ÿà¬¾à¬¨à¬¬à­‡", "à¬¤à­‡à­Ÿà¬¾à¬¨à¬¬à­‡", "à¬šà¬‰à¬°à¬¾à¬¨à¬¬à­‡", "à¬ªà¬žà­à¬šà¬¾à¬¨à¬¬à­‡", "à¬›à­Ÿà¬¾à¬¨à¬¬à­‡", "à¬¸à¬¤à¬¾à¬¨à¬¬à­‡", "à¬…à¬ à¬¾à¬¨à¬¬à­‡", "à¬…à¬¨à­‡à¬¶à¬¤",
    ]
    HUNDRED = "à¬¶à¬¹"
    THOUSAND = "à¬¹à¬œà¬¾à¬°"
    LAKH = "à¬²à¬•à­à¬·"
    CRORE = "à¬•à­‹à¬Ÿà¬¿"

    LETTERS = {
        "A": "à¬", "B": "à¬¬à¬¿", "C": "à¬¸à¬¿", "D": "à¬¡à¬¿", "E": "à¬‡", "F": "à¬à¬«à­",
        "G": "à¬œà¬¿", "H": "à¬à¬šà­", "I": "à¬†à¬‡", "J": "à¬œà­‡", "K": "à¬•à­‡", "L": "à¬à¬²à­",
        "M": "à¬à¬®à­", "N": "à¬à¬¨à­", "O": "à¬“", "P": "à¬ªà¬¿", "Q": "à¬•à­à­Ÿà­", "R": "à¬†à¬°à­",
        "S": "à¬à¬¸à­", "T": "à¬Ÿà¬¿", "U": "à­Ÿà­", "V": "à¬­à¬¿", "W": "à¬¡à¬¬à­à¬²à­à­Ÿà­",
        "X": "à¬à¬•à­à¬¸", "Y": "à­±à¬¾à¬‡", "Z": "à¬œà­‡à¬¡à­",
    }

    MONTHS = ["à¬œà¬¾à¬¨à­à¬†à¬°à­€", "à¬«à­‡à¬¬à­ƒà¬†à¬°à­€", "à¬®à¬¾à¬°à­à¬šà­à¬š", "à¬…à¬ªà­à¬°à­‡à¬²", "à¬®à¬‡", "à¬œà­à¬¨",
              "à¬œà­à¬²à¬¾à¬‡", "à¬…à¬—à¬·à­à¬Ÿ", "à¬¸à­‡à¬ªà­à¬Ÿà­‡à¬®à­à¬¬à¬°", "à¬…à¬•à­à¬Ÿà­‹à¬¬à¬°", "à¬¨à¬­à­‡à¬®à­à¬¬à¬°", "à¬¡à¬¿à¬¸à­‡à¬®à­à¬¬à¬°"]

    SYM = {
        "rupee": "à¬Ÿà¬™à­à¬•à¬¾", "dollar": "à¬¡à¬²à¬¾à¬°", "euro": "à­Ÿà­à¬°à­‹", "pound": "à¬ªà¬¾à¬‰à¬£à­à¬¡",
        "cent": "à¬¸à­‡à¬£à­à¬Ÿ",
        "percent": "à¬ªà­à¬°à¬¤à¬¿à¬¶à¬¤",
        "point": "à¬¦à¬¶à¬®à¬¿à¬•",
        "minus": "à¬®à¬¾à¬‡à¬¨à¬¸à­", "plus": "à¬ªà­à¬²à¬¸à­", "equals": "à¬¸à¬®à¬¾à¬¨",
        "times": "à¬—à­à¬£", "divide": "à¬­à¬¾à¬—",
        "squared": "à¬° à¬¬à¬°à­à¬—", "sqrt": "à¬¬à¬°à­à¬—à¬®à­‚à¬³",
        "lt": "à¬ à¬¾à¬°à­ à¬•à¬®à­", "gt": "à¬ à¬¾à¬°à­ à¬…à¬§à¬¿à¬•",
        "plus_minus": "à¬ªà­à¬²à¬¸à­-à¬®à¬¾à¬‡à¬¨à¬¸à­",
        "at": "à¬†à¬Ÿà­ à¬¦à¬¿ à¬°à­‡à¬Ÿà­", "underscore": "à¬…à¬£à­à¬¡à¬°à¬¸à­à¬•à­‹à¬°", "dot": "à¬¡à¬Ÿà­",
        "slash": "à¬¸à­à¬²à¬¾à¬¸à­", "hash": "à¬¹à­à­Ÿà¬¾à¬¸à­", "amp": "à¬†à¬£à­à¬¡",
        "half": "à¬…à¬§à¬¾", "quarter": "à¬à¬• à¬šà¬¤à­à¬°à­à¬¥à¬¾à¬‚à¬¶", "three_quarters": "à¬¤à¬¿à¬¨à¬¿ à¬šà¬¤à­à¬°à­à¬¥à¬¾à¬‚à¬¶",
        "degree": "à¬¡à¬¿à¬—à­à¬°à­€", "celsius": "à¬¡à¬¿à¬—à­à¬°à­€ à¬¸à­‡à¬²à¬¸à¬¿à­Ÿà¬¸à­", "fahrenheit": "à¬¡à¬¿à¬—à­à¬°à­€ à¬«à¬¾à¬°à­‡à¬¨à¬¹à¬¾à¬‡à¬Ÿà­",
        "minute": "à¬®à¬¿à¬¨à¬¿à¬Ÿà­",
        "paisa": "à¬ªà¬‡à¬¸à¬¾", "per": "à¬ªà­à¬°à¬¤à¬¿", "second": "à¬¸à­‡à¬•à­‡à¬£à­à¬¡",
    }

    TIME_ON_HOUR = "{h}à¬Ÿà¬¾"
    TIME_HM = "{h}à¬Ÿà¬¾ {m} à¬®à¬¿à¬¨à¬¿à¬Ÿà­"
    RANGE = "{a} à¬°à­ {b}"
    RANGE_TO = "{a} à¬°à­ {b} à¬ªà¬°à­à¬¯à­à­Ÿà¬¨à­à¬¤"
    RATIO = "{a} à¬…à¬¨à­à¬ªà¬¾à¬¤ {b}"
    POWER = "{base} à¬° à¬˜à¬¾à¬¤ {exp}"
    FRACTION = "{num} à¬¬à¬Ÿà¬¾ {den}"
    DECADE_SHORT = "{tens} à¬¦à¬¶à¬•"
    DECADE_FULL = "{tens} à¬¦à¬¶à¬•"

    PERIODS = {"morning": "à¬¸à¬•à¬¾à¬³", "afternoon": "à¬¦à¬¿à¬¨",
               "evening": "à¬¸à¬¨à­à¬§à­à­Ÿà¬¾", "night": "à¬°à¬¾à¬¤à¬¿"}

    TIME_MARKERS = ["à¬Ÿà¬¾", "à¬Ÿà¬¾à¬°à­‡", "à¬Ÿà¬¾à¬°à­", "à¬¬à­‡à¬³à­‡", "à¬¬à¬œà­‡"]
    PERIOD_DETECT = {"à¬¸à¬•à¬¾à¬³": "morning", "à¬¸à¬•à¬¾à¬³à­‡": "morning", "à¬¸à¬•à¬¾à¬³à­": "morning",
                     "à¬­à­‹à¬°": "morning", "à¬­à­‹à¬°à¬°à­": "morning",
                     "à¬¦à¬¿à¬¨": "afternoon", "à¬…à¬ªà¬°à¬¾à¬¹à­à¬¨": "afternoon",
                     "à¬®à¬§à­à­Ÿà¬¾à¬¹à­à¬¨": "afternoon", "à¬¦à­à­±à¬¿à¬ªà¬¹à¬°": "afternoon",
                     "à¬¸à¬¨à­à¬§à­à­Ÿà¬¾": "evening", "à¬¸à¬¨à­à¬§à­à­Ÿà¬¾à¬°à­‡": "evening",
                     "à¬°à¬¾à¬¤à¬¿": "night", "à¬°à¬¾à¬¤à¬¿à¬°à­‡": "night", "à¬°à¬¾à¬¤à­à¬°à¬¿": "night"}
    YEAR_WORDS = ["à¬®à¬¸à¬¿à¬¹à¬¾", "à¬®à¬¸à¬¿à¬¹à¬¾à¬°à­‡", "à¬¸à¬¾à¬²", "à¬¸à¬¨", "à¬¬à¬°à­à¬·", "à¬¬à¬°à­à¬·à¬°à­‡"]
    ID_KEYWORDS = ["à¬¨à¬®à­à¬¬à¬°", "à¬¨à¬®à­à¬¬à¬°à­", "à¬–à¬¾à¬¤à¬¾", "à¬†à¬•à¬¾à¬‰à¬£à­à¬Ÿ", "à¬•à­‹à¬¡à­", "à¬…à¬°à­à¬¡à¬°", "à¬«à­‹à¬¨à­", "à¬«à­‹à¬¨", "à¬®à­‹à¬¬à¬¾à¬‡à¬²à­", "à¬®à­‹à¬¬à¬¾à¬‡à¬²", "à¬“à¬Ÿà¬¿à¬ªà¬¿", "à¬ªà¬¿à¬¨à­", "à¬†à¬‡à¬¡à¬¿", "à¬ªà¬¾à¬¸à­±à¬¾à¬°à­à¬¡", "à¬à¬•à­à¬¸à¬ªà­à¬°à­‡à¬¸", "à¬ªà¬¿à¬à¬¨à­â€Œà¬†à¬°à­", "à¬•à¬¾à¬°à­à¬¡", "à¬°à­‡à¬«à¬°à­‡à¬¨à­à¬¸", "à¬Ÿà¬¿à¬•à­‡à¬Ÿ", "à¬†à¬•à¬¾à¬‰à¬£à­à¬Ÿà¬°à­", "à¬Ÿà­à¬°à­‡à¬¨à­", "à­Ÿà­à¬œà¬°", "à¬•à¬¿à¬¸à­à¬¤à¬¿", "à¬…à¬™à­à¬•", "à¬²à¬¾à¬£à¬¦à­‡à¬£"]
    ID_AFTER_WORDS = ["à¬¡à¬¾à¬à¬²", "à¬•à¬²", "à¬¯à­‹à¬—à¬¾à¬¯à­‹à¬—"]
    NUMBER_WORD = "à¬¨à¬®à­à¬¬à¬°"
    ABBREV_UNITS = {"MINS":"à¬®à¬¿à¬¨à¬¿à¬Ÿà­", "MIN":"à¬®à¬¿à¬¨à¬¿à¬Ÿà­", "HRS":"à¬˜à¬£à­à¬Ÿà¬¾", "HR":"à¬˜à¬£à­à¬Ÿà¬¾", "SECS":"à¬¸à­‡à¬•à­‡à¬£à­à¬¡", "SEC":"à¬¸à­‡à¬•à­‡à¬£à­à¬¡", "KMS":"à¬•à¬¿à¬²à­‹à¬®à¬¿à¬Ÿà¬°"}
    LABEL_WORDS = ["à¬…à¬¨à­à¬šà­à¬›à­‡à¬¦", "à¬§à¬¾à¬°à¬¾", "à¬°à­à¬®à­", "à¬•à­‹à¬ à¬°à¬¿", "à­±à¬¾à¬°à­à¬¡",
                   "à¬…à¬§à­à­Ÿà¬¾à­Ÿ", "à¬­à¬¾à¬—", "à¬—à­‡à¬Ÿà­", "à¬•à¬¾à¬‰à¬£à­à¬Ÿà¬°", "à¬¶à­à¬°à­‡à¬£à­€"]
    MINUTE_WORDS = ["à¬®à¬¿à¬¨à¬¿à¬Ÿà­", "à¬®à¬¿à¬¨à¬¿à¬Ÿ"]
    SCALE_WORDS = {"à¬•à­‹à¬Ÿà¬¿": "crore", "à¬²à¬•à­à¬·": "lakh", "à¬²à¬¾à¬–": "lakh",
                   "à¬¹à¬œà¬¾à¬°": "thousand"}
    CURRENCY_WORDS = {"à¬Ÿà¬™à­à¬•à¬¾": "rupee", "à¬Ÿà¬™à­à¬•à¬¾à¬°": "rupee", "à¬Ÿà¬™à­à¬•à¬¾à¬°à­‡": "rupee",
                      "à¬Ÿà¬‚à¬•à¬¾": "rupee", "à¬°à­à¬ªà¬¿": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "à¬ªà­à¬°à¬¥à¬®", 2: "à¬¦à­à­±à¬¿à¬¤à­€à­Ÿ", 3: "à¬¤à­ƒà¬¤à­€à­Ÿ", 4: "à¬šà¬¤à­à¬°à­à¬¥",
                          5: "à¬ªà¬žà­à¬šà¬®", 6: "à¬·à¬·à­à¬ ", 7: "à¬¸à¬ªà­à¬¤à¬®", 8: "à¬…à¬·à­à¬Ÿà¬®",
                          9: "à¬¨à¬¬à¬®", 10: "à¬¦à¬¶à¬®"}
    ORDINAL_SUFFIX = "à¬¤à¬®"
    ORDINAL_INPUT_SUFFIXES = ["à¬¤à¬®"]

    # Odia postpositions agglutinate onto noun-words (à¬Ÿà¬™à­à¬•à¬¾ + à¬°à­ -> à¬Ÿà¬™à­à¬•à¬¾à¬°à­)
    _ATTACH_NOUNS = {"à¬Ÿà¬™à­à¬•à¬¾", "à¬ªà¬‡à¬¸à¬¾", "à¬¡à¬²à¬¾à¬°", "à¬ªà­à¬°à¬¤à¬¿à¬¶à¬¤", "à¬•à¬¿à¬²à­‹à¬®à¬¿à¬Ÿà¬°",
                     "à¬•à¬¿à¬²à­‹à¬—à­à¬°à¬¾à¬®", "à¬®à¬¿à¬Ÿà¬°", "à¬²à¬¿à¬Ÿà¬°", "à¬—à­à¬°à¬¾à¬®", "à¬¡à¬¿à¬—à­à¬°à­€",
                     "à¬®à¬¿à¬¨à¬¿à¬Ÿà­", "à¬˜à¬£à­à¬Ÿà¬¾", "à¬¸à­‡à¬•à­‡à¬£à­à¬¡"}
    _PP = {"à¬°à­‡", "à¬°à­", "à¬•à­", "à¬°", "à¬ à¬¾à¬°à­", "à¬ªà¬°à­à¬¯à­à­Ÿà¬¨à­à¬¤", "à¬®à¬§à­à­Ÿà¬°à­‡",
           "à¬ªà¬¾à¬‡à¬", "à¬…à¬ªà­‡à¬•à­à¬·à¬¾", "à¬­à¬¿à¬¤à¬°à­‡"}

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        if t == last and last in self._ATTACH_NOUNS:
            return rep
        if t in self._PP and last in self._ATTACH_NOUNS:
            return rep + t
        return None


class Punjabi(LangBase):
    code = "pa"
    name = "Punjabi"
    native_digits = "à©¦à©§à©¨à©©à©ªà©«à©¬à©­à©®à©¯"
    SINGULAR_NOUNS = {"à¨˜à©°à¨Ÿà©‡": "à¨˜à©°à¨Ÿà¨¾"}

    ONES = [
        "à¨¸à¨¿à¨«à¨¼à¨°", "à¨‡à©±à¨•", "à¨¦à©‹", "à¨¤à¨¿à©°à¨¨", "à¨šà¨¾à¨°", "à¨ªà©°à¨œ", "à¨›à©‡", "à¨¸à©±à¨¤", "à¨…à©±à¨ ", "à¨¨à©Œà¨‚",
        "à¨¦à¨¸", "à¨—à¨¿à¨†à¨°à¨¾à¨‚", "à¨¬à¨¾à¨°à¨¾à¨‚", "à¨¤à©‡à¨°à¨¾à¨‚", "à¨šà©Œà¨¦à¨¾à¨‚", "à¨ªà©°à¨¦à¨°à¨¾à¨‚", "à¨¸à©‹à¨²à¨¾à¨‚", "à¨¸à¨¤à¨¾à¨°à¨¾à¨‚", "à¨…à¨ à¨¾à¨°à¨¾à¨‚", "à¨‰à©±à¨¨à©€",
        "à¨µà©€à¨¹", "à¨‡à©±à¨•à©€", "à¨¬à¨¾à¨ˆ", "à¨¤à©‡à¨ˆ", "à¨šà©Œà¨µà©€", "à¨ªà©±à¨šà©€", "à¨›à©±à¨¬à©€", "à¨¸à¨¤à¨¾à¨ˆ", "à¨…à¨ à¨¾à¨ˆ", "à¨‰à¨£à©±à¨¤à©€",
        "à¨¤à©€à¨¹", "à¨‡à¨•à©±à¨¤à©€", "à¨¬à©±à¨¤à©€", "à¨¤à©‡à¨¤à©€", "à¨šà©Œà¨‚à¨¤à©€", "à¨ªà©ˆà¨‚à¨¤à©€", "à¨›à©±à¨¤à©€", "à¨¸à©ˆà¨‚à¨¤à©€", "à¨…à¨ à©±à¨¤à©€", "à¨‰à¨£à¨¤à¨¾à¨²à©€",
        "à¨šà¨¾à¨²à©€", "à¨‡à¨•à¨¤à¨¾à¨²à©€", "à¨¬à¨¿à¨†à¨²à©€", "à¨¤à¨°à¨¤à¨¾à¨²à©€", "à¨šà©à¨¤à¨¾à¨²à©€", "à¨ªà©°à¨¤à¨¾à¨²à©€", "à¨›à¨¿à¨†à¨²à©€", "à¨¸à©°à¨¤à¨¾à¨²à©€", "à¨…à¨ à¨¤à¨¾à¨²à©€", "à¨‰à¨£à©°à¨œà¨¾",
        "à¨ªà©°à¨œà¨¾à¨¹", "à¨‡à¨•à¨µà©°à¨œà¨¾", "à¨¬à¨µà©°à¨œà¨¾", "à¨¤à¨°à¨µà©°à¨œà¨¾", "à¨šà©à¨°à©°à¨œà¨¾", "à¨ªà¨šà¨µà©°à¨œà¨¾", "à¨›à¨ªà©°à¨œà¨¾", "à¨¸à¨¤à¨µà©°à¨œà¨¾", "à¨…à¨ à¨µà©°à¨œà¨¾", "à¨‰à¨£à¨¾à¨¹à¨ ",
        "à¨¸à©±à¨ ", "à¨‡à¨•à¨¾à¨¹à¨ ", "à¨¬à¨¾à¨¹à¨ ", "à¨¤à©à¨°à©‡à¨¹à¨ ", "à¨šà©Œà¨‚à¨¹à¨ ", "à¨ªà©ˆà¨‚à¨¹à¨ ", "à¨›à¨¿à¨†à¨¹à¨ ", "à¨¸à¨¤à¨¾à¨¹à¨ ", "à¨…à¨ à¨¾à¨¹à¨ ", "à¨‰à¨£à©±à¨¤à¨°",
        "à¨¸à©±à¨¤à¨°", "à¨‡à¨•à¨¹à©±à¨¤à¨°", "à¨¬à¨¹à©±à¨¤à¨°", "à¨¤à¨¿à¨¹à©±à¨¤à¨°", "à¨šà©à¨¹à©±à¨¤à¨°", "à¨ªà©°à¨à©±à¨¤à¨°", "à¨›à¨¿à¨¹à©±à¨¤à¨°", "à¨¸à¨¤à©±à¨¤à¨°", "à¨…à¨ à©±à¨¤à¨°", "à¨‰à¨£à¨¾à¨¸à©€",
        "à¨…à©±à¨¸à©€", "à¨‡à¨•à¨¾à¨¸à©€", "à¨¬à¨¿à¨†à¨¸à©€", "à¨¤à¨°à¨¾à¨¸à©€", "à¨šà©à¨°à¨¾à¨¸à©€", "à¨ªà¨šà¨¾à¨¸à©€", "à¨›à¨¿à¨†à¨¸à©€", "à¨¸à¨¤à¨¾à¨¸à©€", "à¨…à¨ à¨¾à¨¸à©€", "à¨‰à¨£à¨¾à¨¨à¨µà©‡à¨‚",
        "à¨¨à©±à¨¬à©‡", "à¨‡à¨•à¨¾à¨¨à¨µà©‡à¨‚", "à¨¬à¨¾à¨¨à¨µà©‡à¨‚", "à¨¤à¨°à¨¾à¨¨à¨µà©‡à¨‚", "à¨šà©à¨°à¨¾à¨¨à¨µà©‡à¨‚", "à¨ªà¨šà¨¾à¨¨à¨µà©‡à¨‚", "à¨›à¨¿à¨†à¨¨à¨µà©‡à¨‚", "à¨¸à¨¤à¨¾à¨¨à¨µà©‡à¨‚", "à¨…à¨ à¨¾à¨¨à¨µà©‡à¨‚", "à¨¨à©œà¨¿à©°à¨¨à¨µà©‡à¨‚",
    ]
    HUNDRED = "à¨¸à©Œ"
    THOUSAND = "à¨¹à¨œà¨¼à¨¾à¨°"
    LAKH = "à¨²à©±à¨–"
    CRORE = "à¨•à¨°à©‹à©œ"

    LETTERS = {
        "A": "à¨", "B": "à¨¬à©€", "C": "à¨¸à©€", "D": "à¨¡à©€", "E": "à¨ˆ", "F": "à¨à©±à¨«",
        "G": "à¨œà©€", "H": "à¨à©±à¨š", "I": "à¨†à¨ˆ", "J": "à¨œà©‡", "K": "à¨•à©‡", "L": "à¨à©±à¨²",
        "M": "à¨à©±à¨®", "N": "à¨à©±à¨¨", "O": "à¨“", "P": "à¨ªà©€", "Q": "à¨•à¨¿à¨Š", "R": "à¨†à¨°",
        "S": "à¨à©±à¨¸", "T": "à¨Ÿà©€", "U": "à¨¯à©‚", "V": "à¨µà©€", "W": "à¨¡à¨¬à¨²à¨¿à¨Š",
        "X": "à¨à¨•à¨¸", "Y": "à¨µà¨¾à¨ˆ", "Z": "à¨œà¨¼à©ˆà©±à¨¡",
    }

    MONTHS = ["à¨œà¨¨à¨µà¨°à©€", "à¨«à¨¼à¨°à¨µà¨°à©€", "à¨®à¨¾à¨°à¨š", "à¨…à¨ªà©à¨°à©ˆà¨²", "à¨®à¨ˆ", "à¨œà©‚à¨¨",
              "à¨œà©à¨²à¨¾à¨ˆ", "à¨…à¨—à¨¸à¨¤", "à¨¸à¨¤à©°à¨¬à¨°", "à¨…à¨•à¨¤à©‚à¨¬à¨°", "à¨¨à¨µà©°à¨¬à¨°", "à¨¦à¨¸à©°à¨¬à¨°"]

    SYM = {
        "rupee": "à¨°à©à¨ªà¨", "dollar": "à¨¡à¨¾à¨²à¨°", "euro": "à¨¯à©‚à¨°à©‹", "pound": "à¨ªà©Œà¨‚à¨¡",
        "cent": "à¨¸à©ˆà¨‚à¨Ÿ",
        "percent": "à¨«à¨¼à©€à¨¸à¨¦à©€",
        "point": "à¨¦à¨¸à¨¼à¨®à¨²à¨µ",
        "minus": "à¨®à¨¾à¨ˆà¨¨à¨¸", "plus": "à¨ªà¨²à©±à¨¸", "equals": "à¨¬à¨°à¨¾à¨¬à¨°",
        "times": "à¨—à©à¨£à¨¾", "divide": "à¨­à¨¾à¨—",
        "squared": "à¨¦à¨¾ à¨µà¨°à¨—", "sqrt": "à¨µà¨°à¨—à¨®à©‚à¨²",
        "lt": "à¨¤à©‹à¨‚ à¨˜à©±à¨Ÿ", "gt": "à¨¤à©‹à¨‚ à¨µà©±à¨§",
        "plus_minus": "à¨§à¨¨-à¨°à¨¿à¨£",
        "at": "à¨à¨Ÿ à¨¦ à¨°à©‡à¨Ÿ", "underscore": "à¨…à©°à¨¡à¨°à¨¸à¨•à©‹à¨°", "dot": "à¨¡à¨¾à¨Ÿ",
        "slash": "à¨¸à¨²à©ˆà¨¸à¨¼", "hash": "à¨¹à©ˆà¨¸à¨¼", "amp": "à¨…à¨¤à©‡",
        "half": "à¨…à©±à¨§à¨¾", "quarter": "à¨šà©Œà¨¥à¨¾à¨ˆ", "three_quarters": "à¨¤à¨¿à©°à¨¨ à¨šà©Œà¨¥à¨¾à¨ˆ",
        "degree": "à¨¡à¨¿à¨—à¨°à©€", "celsius": "à¨¡à¨¿à¨—à¨°à©€ à¨¸à©ˆà¨²à¨¸à©€à¨…à¨¸", "fahrenheit": "à¨¡à¨¿à¨—à¨°à©€ à¨«à¨¾à¨°à¨¨à¨¹à¨¾à¨ˆà¨Ÿ",
        "minute": "à¨®à¨¿à©°à¨Ÿ", "second": "à¨¸à¨•à¨¿à©°à¨Ÿ",
    }

    TIME_ON_HOUR = "{h} à¨µà¨œà©‡"
    TIME_HM = "{h} à¨µà©±à¨œ à¨•à©‡ {m} à¨®à¨¿à©°à¨Ÿ"
    RANGE = "{a} à¨¤à©‹à¨‚ {b}"
    RANGE_TO = "{a} à¨¤à©‹à¨‚ {b} à¨¤à©±à¨•"
    RATIO = "{a} à¨…à¨¨à©à¨ªà¨¾à¨¤ {b}"
    POWER = "{base} à¨¦à©€ à¨˜à¨¾à¨¤ {exp}"
    FRACTION = "{num} à¨¬à¨Ÿà¨¾ {den}"
    DECADE_SHORT = "{tens} à¨¦à©‡"
    DECADE_FULL = "{tens} à¨¦à¨¾ à¨¦à¨¹à¨¾à¨•à¨¾"

    PERIODS = {"morning": "à¨¸à¨µà©‡à¨°à©‡", "afternoon": "à¨¦à©à¨ªà¨¹à¨¿à¨°", "evening": "à¨¸à¨¼à¨¾à¨®", "night": "à¨°à¨¾à¨¤"}

    TIME_MARKERS = ["à¨µà¨œà©‡", "à¨µà©±à¨œ à¨•à©‡", "à¨µà¨œà¨•à©‡", "à¨µà©±à¨œà©‡"]
    PERIOD_DETECT = {"à¨¸à¨µà©‡à¨°à©‡": "morning", "à¨¸à¨µà©‡à¨°": "morning", "à¨¤à©œà¨•à©‡": "morning",
                     "à¨¦à©à¨ªà¨¹à¨¿à¨°": "afternoon", "à¨¸à¨¼à¨¾à¨®": "evening", "à¨¸à¨¼à¨¾à¨®à©€à¨‚": "evening",
                     "à¨°à¨¾à¨¤": "night", "à¨°à¨¾à¨¤à©€à¨‚": "night"}
    YEAR_WORDS = ["à¨¸à¨¾à¨²", "à¨µà¨°à©à¨¹à¨¾", "à¨µà¨°à©à¨¹à©‡", "à¨¸à©°à¨¨"]
    ID_KEYWORDS = ["à¨¨à©°à¨¬à¨°", "à¨–à¨¾à¨¤à¨¾", "à¨–à¨¾à¨¤à©‡", "à¨•à©‹à¨¡", "à¨†à¨°à¨¡à¨°", "à¨«à¨¼à©‹à¨¨", "à¨«à©‹à¨¨", "à¨®à©‹à¨¬à¨¾à¨ˆà¨²", "à¨®à©‹à¨¬à¨¾à¨‡à¨²", "à¨…à¨•à¨¾à¨Šà¨‚à¨Ÿ", "à¨…à¨•à¨¾à¨Šà¨‚à¨Ÿà¨µà¨¿à©±à¨š", "à¨ªà¨¿à©°à¨¨", "à¨†à¨ˆà¨¡à©€", "à¨ªà¨¾à¨¸à¨µà¨°à¨¡", "à¨Ÿà©à¨°à©‡à¨¨", "à¨à¨•à¨¸à¨ªà©à¨°à©ˆà¨¸", "à¨ªà©€à¨à¨¨à¨†à¨°", "à¨•à¨¾à¨°à¨¡", "à¨°à©ˆà¨«à¨°à©ˆà¨‚à¨¸", "à¨Ÿà¨¿à¨•à¨Ÿ", "à¨“à¨Ÿà©€à¨ªà©€", "à¨¯à©‚à¨œà¨¼à¨°", "à¨•à¨¿à¨¸à¨¼à¨¤", "à¨…à©°à¨•", "à¨²à©ˆà¨£-à¨¦à©‡à¨£"]
    ID_AFTER_WORDS = ["à¨¡à¨¾à¨‡à¨²", "à¨•à¨¾à¨²", "à¨¸à©°à¨ªà¨°à¨•"]
    NUMBER_WORD = "à¨¨à©°à¨¬à¨°"
    ABBREV_UNITS = {"MINS":"à¨®à¨¿à©°à¨Ÿ", "MIN":"à¨®à¨¿à©°à¨Ÿ", "HRS":"à¨˜à©°à¨Ÿà©‡", "HR":"à¨˜à©°à¨Ÿà¨¾", "SECS":"à¨¸à¨•à¨¿à©°à¨Ÿ", "SEC":"à¨¸à¨•à¨¿à©°à¨Ÿ", "KMS":"à¨•à¨¿à¨²à©‹à¨®à©€à¨Ÿà¨°"}
    LABEL_WORDS = ["à¨§à¨¾à¨°à¨¾", "à¨…à¨¨à©à¨›à©‡à¨¦", "à¨•à¨®à¨°à¨¾", "à¨•à¨²à¨¾à¨¸", "à¨µà¨¾à¨°à¨¡", "à¨…à¨§à¨¿à¨†à¨‡", "à¨­à¨¾à¨—", "à¨œà¨®à¨¾à¨¤"]
    MINUTE_WORDS = ["à¨®à¨¿à©°à¨Ÿ"]
    SCALE_WORDS = {"à¨•à¨°à©‹à©œ": "crore", "à¨²à©±à¨–": "lakh", "à¨¹à¨œà¨¼à¨¾à¨°": "thousand", "à¨¹à¨œà¨¾à¨°": "thousand",
                   "à¨…à¨°à¨¬": "arab"}
    CURRENCY_WORDS = {"à¨°à©à¨ªà¨": "rupee", "à¨°à©à¨ªà¨¿à¨†": "rupee", "à¨°à©à¨ªà¨ˆà¨": "rupee", "à¨°à©à¨ªà¨¿à¨": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "à¨ªà¨¹à¨¿à¨²à¨¾", 2: "à¨¦à©‚à¨œà¨¾", 3: "à¨¤à©€à¨œà¨¾", 4: "à¨šà©Œà¨¥à¨¾", 6: "à¨›à©‡à¨µà¨¾à¨‚"}
    ORDINAL_SUFFIX = "à¨µà¨¾à¨‚"
    ORDINAL_INPUT_SUFFIXES = ["à¨µà©€à¨‚", "à¨µà¨¾à¨‚", "à¨µà©‡à¨‚", "à¨µà¨¾"]

    def _ordinal_compose(self, n, src_suffix):
        # 12à¨µà©€à¨‚ -> à¨¬à¨¾à¨°à¨¾à¨‚à¨µà©€à¨‚ : cardinal + preserved suffix
        if src_suffix:
            return f"{self.cardinal(n)}{src_suffix}"
        if n in self.ORDINAL_EXCEPTIONS:
            return self.ORDINAL_EXCEPTIONS[n]
        return f"{self.cardinal(n)}{self.ORDINAL_SUFFIX}"


class Tamil(LangBase):
    code = "ta"
    name = "Tamil"
    native_digits = "à¯¦à¯§à¯¨à¯©à¯ªà¯«à¯¬à¯­à¯®à¯¯"
    COUNT_ONE = "à®’à®°à¯"

    ONES = [
        "à®ªà¯‚à®œà¯à®œà®¿à®¯à®®à¯", "à®’à®©à¯à®±à¯", "à®‡à®°à®£à¯à®Ÿà¯", "à®®à¯‚à®©à¯à®±à¯", "à®¨à®¾à®©à¯à®•à¯",
        "à®à®¨à¯à®¤à¯", "à®†à®±à¯", "à®à®´à¯", "à®Žà®Ÿà¯à®Ÿà¯", "à®’à®©à¯à®ªà®¤à¯",
        "à®ªà®¤à¯à®¤à¯", "à®ªà®¤à®¿à®©à¯Šà®©à¯à®±à¯", "à®ªà®©à¯à®©à®¿à®°à®£à¯à®Ÿà¯", "à®ªà®¤à®¿à®©à¯à®®à¯‚à®©à¯à®±à¯", "à®ªà®¤à®¿à®©à®¾à®©à¯à®•à¯",
        "à®ªà®¤à®¿à®©à¯ˆà®¨à¯à®¤à¯", "à®ªà®¤à®¿à®©à®¾à®±à¯", "à®ªà®¤à®¿à®©à¯‡à®´à¯", "à®ªà®¤à®¿à®©à¯†à®Ÿà¯à®Ÿà¯", "à®ªà®¤à¯à®¤à¯Šà®©à¯à®ªà®¤à¯",
        "à®‡à®°à¯à®ªà®¤à¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®±à¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à®¿à®°à®£à¯à®Ÿà¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à¯à®®à¯‚à®©à¯à®±à¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à¯à®¨à®¾à®©à¯à®•à¯",
        "à®‡à®°à¯à®ªà®¤à¯à®¤à¯ˆà®¨à¯à®¤à¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à®¾à®±à¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à¯‡à®´à¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à¯†à®Ÿà¯à®Ÿà¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®ªà®¤à¯",
        "à®®à¯à®ªà¯à®ªà®¤à¯", "à®®à¯à®ªà¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®±à¯", "à®®à¯à®ªà¯à®ªà®¤à¯à®¤à®¿à®°à®£à¯à®Ÿà¯", "à®®à¯à®ªà¯à®ªà®¤à¯à®¤à¯à®®à¯‚à®©à¯à®±à¯", "à®®à¯à®ªà¯à®ªà®¤à¯à®¤à¯à®¨à®¾à®©à¯à®•à¯",
        "à®®à¯à®ªà¯à®ªà®¤à¯à®¤à¯ˆà®¨à¯à®¤à¯", "à®®à¯à®ªà¯à®ªà®¤à¯à®¤à®¾à®±à¯", "à®®à¯à®ªà¯à®ªà®¤à¯à®¤à¯‡à®´à¯", "à®®à¯à®ªà¯à®ªà®¤à¯à®¤à¯†à®Ÿà¯à®Ÿà¯", "à®®à¯à®ªà¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®ªà®¤à¯",
        "à®¨à®¾à®±à¯à®ªà®¤à¯", "à®¨à®¾à®±à¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®±à¯", "à®¨à®¾à®±à¯à®ªà®¤à¯à®¤à®¿à®°à®£à¯à®Ÿà¯", "à®¨à®¾à®±à¯à®ªà®¤à¯à®¤à¯à®®à¯‚à®©à¯à®±à¯", "à®¨à®¾à®±à¯à®ªà®¤à¯à®¤à¯à®¨à®¾à®©à¯à®•à¯",
        "à®¨à®¾à®±à¯à®ªà®¤à¯à®¤à¯ˆà®¨à¯à®¤à¯", "à®¨à®¾à®±à¯à®ªà®¤à¯à®¤à®¾à®±à¯", "à®¨à®¾à®±à¯à®ªà®¤à¯à®¤à¯‡à®´à¯", "à®¨à®¾à®±à¯à®ªà®¤à¯à®¤à¯†à®Ÿà¯à®Ÿà¯", "à®¨à®¾à®±à¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®ªà®¤à¯",
        "à®à®®à¯à®ªà®¤à¯", "à®à®®à¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®±à¯", "à®à®®à¯à®ªà®¤à¯à®¤à®¿à®°à®£à¯à®Ÿà¯", "à®à®®à¯à®ªà®¤à¯à®¤à¯à®®à¯‚à®©à¯à®±à¯", "à®à®®à¯à®ªà®¤à¯à®¤à¯à®¨à®¾à®©à¯à®•à¯",
        "à®à®®à¯à®ªà®¤à¯à®¤à¯ˆà®¨à¯à®¤à¯", "à®à®®à¯à®ªà®¤à¯à®¤à®¾à®±à¯", "à®à®®à¯à®ªà®¤à¯à®¤à¯‡à®´à¯", "à®à®®à¯à®ªà®¤à¯à®¤à¯†à®Ÿà¯à®Ÿà¯", "à®à®®à¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®ªà®¤à¯",
        "à®…à®±à¯à®ªà®¤à¯", "à®…à®±à¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®±à¯", "à®…à®±à¯à®ªà®¤à¯à®¤à®¿à®°à®£à¯à®Ÿà¯", "à®…à®±à¯à®ªà®¤à¯à®¤à¯à®®à¯‚à®©à¯à®±à¯", "à®…à®±à¯à®ªà®¤à¯à®¤à¯à®¨à®¾à®©à¯à®•à¯",
        "à®…à®±à¯à®ªà®¤à¯à®¤à¯ˆà®¨à¯à®¤à¯", "à®…à®±à¯à®ªà®¤à¯à®¤à®¾à®±à¯", "à®…à®±à¯à®ªà®¤à¯à®¤à¯‡à®´à¯", "à®…à®±à¯à®ªà®¤à¯à®¤à¯†à®Ÿà¯à®Ÿà¯", "à®…à®±à¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®ªà®¤à¯",
        "à®Žà®´à¯à®ªà®¤à¯", "à®Žà®´à¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®±à¯", "à®Žà®´à¯à®ªà®¤à¯à®¤à®¿à®°à®£à¯à®Ÿà¯", "à®Žà®´à¯à®ªà®¤à¯à®¤à¯à®®à¯‚à®©à¯à®±à¯", "à®Žà®´à¯à®ªà®¤à¯à®¤à¯à®¨à®¾à®©à¯à®•à¯",
        "à®Žà®´à¯à®ªà®¤à¯à®¤à¯ˆà®¨à¯à®¤à¯", "à®Žà®´à¯à®ªà®¤à¯à®¤à®¾à®±à¯", "à®Žà®´à¯à®ªà®¤à¯à®¤à¯‡à®´à¯", "à®Žà®´à¯à®ªà®¤à¯à®¤à¯†à®Ÿà¯à®Ÿà¯", "à®Žà®´à¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®ªà®¤à¯",
        "à®Žà®£à¯à®ªà®¤à¯", "à®Žà®£à¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®±à¯", "à®Žà®£à¯à®ªà®¤à¯à®¤à®¿à®°à®£à¯à®Ÿà¯", "à®Žà®£à¯à®ªà®¤à¯à®¤à¯à®®à¯‚à®©à¯à®±à¯", "à®Žà®£à¯à®ªà®¤à¯à®¤à¯à®¨à®¾à®©à¯à®•à¯",
        "à®Žà®£à¯à®ªà®¤à¯à®¤à¯ˆà®¨à¯à®¤à¯", "à®Žà®£à¯à®ªà®¤à¯à®¤à®¾à®±à¯", "à®Žà®£à¯à®ªà®¤à¯à®¤à¯‡à®´à¯", "à®Žà®£à¯à®ªà®¤à¯à®¤à¯†à®Ÿà¯à®Ÿà¯", "à®Žà®£à¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®ªà®¤à¯",
        "à®¤à¯Šà®£à¯à®£à¯‚à®±à¯", "à®¤à¯Šà®£à¯à®£à¯‚à®±à¯à®±à¯Šà®©à¯à®±à¯", "à®¤à¯Šà®£à¯à®£à¯‚à®±à¯à®±à®¿à®°à®£à¯à®Ÿà¯", "à®¤à¯Šà®£à¯à®£à¯‚à®±à¯à®±à¯à®®à¯‚à®©à¯à®±à¯", "à®¤à¯Šà®£à¯à®£à¯‚à®±à¯à®±à¯à®¨à®¾à®©à¯à®•à¯",
        "à®¤à¯Šà®£à¯à®£à¯‚à®±à¯à®±à¯ˆà®¨à¯à®¤à¯", "à®¤à¯Šà®£à¯à®£à¯‚à®±à¯à®±à®¾à®±à¯", "à®¤à¯Šà®£à¯à®£à¯‚à®±à¯à®±à¯‡à®´à¯", "à®¤à¯Šà®£à¯à®£à¯‚à®±à¯à®±à¯†à®Ÿà¯à®Ÿà¯", "à®¤à¯Šà®£à¯à®£à¯‚à®±à¯à®±à¯Šà®©à¯à®ªà®¤à¯",
    ]
    HUNDRED = "à®¨à¯‚à®±à¯"
    THOUSAND = "à®†à®¯à®¿à®°à®®à¯"
    LAKH = "à®²à®Ÿà¯à®šà®®à¯"
    CRORE = "à®•à¯‹à®Ÿà®¿"

    # Fused hundreds (index 1..9) and their combining forms before a remainder.
    HUNDREDS = ["", "à®¨à¯‚à®±à¯", "à®‡à®°à¯à®¨à¯‚à®±à¯", "à®®à¯à®¨à¯à®¨à¯‚à®±à¯", "à®¨à®¾à®©à¯‚à®±à¯", "à®à®¨à¯à®¨à¯‚à®±à¯",
                "à®…à®±à¯à®¨à¯‚à®±à¯", "à®Žà®´à¯à®¨à¯‚à®±à¯", "à®Žà®£à¯à®£à¯‚à®±à¯", "à®¤à¯Šà®³à¯à®³à®¾à®¯à®¿à®°à®®à¯"]
    HUNDREDS_COMB = ["", "à®¨à¯‚à®±à¯à®±à¯", "à®‡à®°à¯à®¨à¯‚à®±à¯à®±à¯", "à®®à¯à®¨à¯à®¨à¯‚à®±à¯à®±à¯", "à®¨à®¾à®©à¯‚à®±à¯à®±à¯", "à®à®¨à¯à®¨à¯‚à®±à¯à®±à¯",
                     "à®…à®±à¯à®¨à¯‚à®±à¯à®±à¯", "à®Žà®´à¯à®¨à¯‚à®±à¯à®±à¯", "à®Žà®£à¯à®£à¯‚à®±à¯à®±à¯", "à®¤à¯Šà®³à¯à®³à®¾à®¯à®¿à®°à®¤à¯à®¤à¯"]

    LETTERS = {
        "A": "à®", "B": "à®ªà®¿", "C": "à®šà®¿", "D": "à®Ÿà®¿", "E": "à®‡", "F": "à®Žà®ƒà®ªà¯",
        "G": "à®œà®¿", "H": "à®Žà®šà¯", "I": "à®", "J": "à®œà¯‡", "K": "à®•à¯‡", "L": "à®Žà®²à¯",
        "M": "à®Žà®®à¯", "N": "à®Žà®©à¯", "O": "à®“", "P": "à®ªà®¿", "Q": "à®•à®¿à®¯à¯‚", "R": "à®†à®°à¯",
        "S": "à®Žà®¸à¯", "T": "à®Ÿà®¿", "U": "à®¯à¯‚", "V": "à®µà®¿", "W": "à®Ÿà®ªà®¿à®³à¯à®¯à¯‚",
        "X": "à®Žà®•à¯à®¸à¯", "Y": "à®’à®¯à¯", "Z": "à®‡à®šà®Ÿà¯",
    }

    MONTHS = ["à®œà®©à®µà®°à®¿", "à®ªà®¿à®ªà¯à®°à®µà®°à®¿", "à®®à®¾à®°à¯à®šà¯", "à®à®ªà¯à®°à®²à¯", "à®®à¯‡", "à®œà¯‚à®©à¯",
              "à®œà¯‚à®²à¯ˆ", "à®†à®•à®¸à¯à®Ÿà¯", "à®šà¯†à®ªà¯à®Ÿà®®à¯à®ªà®°à¯", "à®…à®•à¯à®Ÿà¯‹à®ªà®°à¯", "à®¨à®µà®®à¯à®ªà®°à¯", "à®Ÿà®¿à®šà®®à¯à®ªà®°à¯"]

    SYM = {
        "rupee": "à®°à¯‚à®ªà®¾à®¯à¯", "dollar": "à®Ÿà®¾à®²à®°à¯", "euro": "à®¯à¯‚à®°à¯‹", "pound": "à®ªà®µà¯à®£à¯à®Ÿà¯",
        "cent": "à®šà¯†à®©à¯à®Ÿà¯",
        "percent": "à®šà®¤à®µà¯€à®¤à®®à¯",
        "point": "à®ªà¯à®³à¯à®³à®¿",
        "minus": "à®®à¯ˆà®©à®¸à¯", "plus": "à®ªà®¿à®³à®¸à¯", "equals": "à®šà®®à®®à¯",
        "times": "à®ªà¯†à®°à¯à®•à¯à®•à®²à¯", "divide": "à®µà®•à¯à®¤à¯à®¤à®²à¯",
        "squared": "à®‡à®©à¯ à®µà®°à¯à®•à¯à®•à®®à¯", "sqrt": "à®µà®°à¯à®•à¯à®•à®®à¯‚à®²à®®à¯",
        "lt": "à®µà®¿à®Ÿà®•à¯ à®•à¯à®±à¯ˆà®µà¯", "gt": "à®µà®¿à®Ÿ à®…à®¤à®¿à®•à®®à¯",
        "plus_minus": "à®ªà®¿à®³à®¸à¯ à®®à¯ˆà®©à®¸à¯",
        "at": "à®…à®Ÿà¯ à®¤à®¿ à®°à¯‡à®Ÿà¯", "underscore": "à®…à®£à¯à®Ÿà®°à¯à®¸à¯à®•à¯‹à®°à¯", "dot": "à®Ÿà®¾à®Ÿà¯",
        "slash": "à®¸à¯à®²à®¾à®·à¯", "hash": "à®¹à¯‡à®·à¯", "amp": "à®®à®±à¯à®±à¯à®®à¯",
        "half": "à®…à®°à¯ˆ", "quarter": "à®•à®¾à®²à¯", "three_quarters": "à®®à¯à®•à¯à®•à®¾à®²à¯",
        "degree": "à®Ÿà®¿à®•à®¿à®°à®¿", "celsius": "à®Ÿà®¿à®•à®¿à®°à®¿ à®šà¯†à®²à¯à®šà®¿à®¯à®¸à¯",
        "fahrenheit": "à®Ÿà®¿à®•à®¿à®°à®¿ à®ƒà®ªà®¾à®°à®©à¯à®¹à¯€à®Ÿà¯",
        "minute": "à®¨à®¿à®®à®¿à®Ÿà®®à¯", "second": "à®µà®¿à®©à®¾à®Ÿà®¿",
    }

    TIME_ON_HOUR = "{h} à®®à®£à®¿"
    TIME_HM = "{h} à®®à®£à®¿ {m} à®¨à®¿à®®à®¿à®Ÿà®®à¯"
    RANGE = "{a} à®®à¯à®¤à®²à¯ {b}"
    RANGE_TO = "{a} à®®à¯à®¤à®²à¯ {b} à®µà®°à¯ˆ"
    RATIO = "{a} à®µà®¿à®•à®¿à®¤à®®à¯ {b}"
    POWER = "{base} à®‡à®©à¯ à®…à®Ÿà¯à®•à¯à®•à¯ {exp}"
    FRACTION = "{num} à®µà®•à¯à®¤à¯à®¤à®²à¯ {den}"
    DECADE_SHORT = "{tens}à®•à®³à¯"
    DECADE_FULL = "{tens}à®•à®³à®¿à®©à¯ à®¤à®šà®¾à®ªà¯à®¤à®®à¯"

    PERIODS = {"morning": "à®•à®¾à®²à¯ˆ", "afternoon": "à®®à®¤à®¿à®¯à®®à¯",
               "evening": "à®®à®¾à®²à¯ˆ", "night": "à®‡à®°à®µà¯"}

    TIME_MARKERS = ["à®®à®£à®¿à®•à¯à®•à¯", "à®®à®£à®¿"]
    MARKER_SUFFIXES = {"à®®à®£à®¿à®•à¯à®•à¯": "à®•à¯à®•à¯"}
    PERIOD_DETECT = {"à®•à®¾à®²à¯ˆ": "morning", "à®…à®¤à®¿à®•à®¾à®²à¯ˆ": "morning",
                     "à®®à®¤à®¿à®¯à®®à¯": "afternoon", "à®¨à®£à¯à®ªà®•à®²à¯": "afternoon",
                     "à®ªà®¿à®±à¯à®ªà®•à®²à¯": "afternoon",
                     "à®®à®¾à®²à¯ˆ": "evening", "à®‡à®°à®µà¯": "night"}
    YEAR_WORDS = ["à®†à®£à¯à®Ÿà¯", "à®†à®£à¯à®Ÿà®¿à®²à¯", "à®µà®°à¯à®Ÿà®®à¯", "à®µà®°à¯à®Ÿà®¤à¯à®¤à®¿à®²à¯", "à®µà®°à¯à®·à®®à¯"]
    ID_KEYWORDS = ["à®Žà®£à¯", "à®¨à®®à¯à®ªà®°à¯", "à®•à®£à®•à¯à®•à¯", "à®¤à¯Šà®²à¯ˆà®ªà¯‡à®šà®¿", "à®ªà¯‹à®©à¯", "à®®à¯Šà®ªà¯ˆà®²à¯", "à®†à®°à¯à®Ÿà®°à¯", "à®“à®Ÿà®¿à®ªà®¿", "à®ªà®¿à®©à¯", "à®à®Ÿà®¿", "à®•à®Ÿà®µà¯à®šà¯à®šà¯Šà®²à¯", "à®°à®¯à®¿à®²à¯", "à®Žà®•à¯à®¸à¯à®ªà®¿à®°à®¸à¯", "à®ªà®¿à®Žà®©à¯à®†à®°à¯", "à®•à®¾à®°à¯à®Ÿà¯", "à®•à¯à®±à®¿à®¯à¯€à®Ÿà¯", "à®Ÿà®¿à®•à¯à®•à¯†à®Ÿà¯", "à®•à®£à®•à¯à®•à®¿à®²à¯", "à®¯à¯‚à®šà®°à¯", "à®¤à®µà®£à¯ˆ", "à®‡à®²à®•à¯à®•à®®à¯", "à®ªà®°à®¿à®µà®°à¯à®¤à¯à®¤à®©à¯ˆ"]
    ID_AFTER_WORDS = ["à®Ÿà®¯à®²à¯", "à®…à®´à¯ˆ", "à®¤à¯Šà®Ÿà®°à¯à®ªà¯"]
    NUMBER_WORD = "à®Žà®£à¯"
    ABBREV_UNITS = {"MINS":"à®¨à®¿à®®à®¿à®Ÿà®®à¯", "MIN":"à®¨à®¿à®®à®¿à®Ÿà®®à¯", "HRS":"à®®à®£à®¿", "HR":"à®®à®£à®¿", "SECS":"à®µà®¿à®¨à®¾à®Ÿà®¿", "SEC":"à®µà®¿à®¨à®¾à®Ÿà®¿", "KMS":"à®•à®¿à®²à¯‹à®®à¯€à®Ÿà¯à®Ÿà®°à¯"}
    LABEL_WORDS = ["à®ªà®¿à®°à®¿à®µà¯", "à®šà®°à®¤à¯à®¤à¯", "à®…à®±à¯ˆ", "à®µà®¾à®°à¯à®Ÿà¯", "à®…à®¤à¯à®¤à®¿à®¯à®¾à®¯à®®à¯",
                   "à®µà®•à¯à®ªà¯à®ªà¯", "à®ªà®¾à®•à®®à¯", "à®•à®µà¯à®£à¯à®Ÿà®°à¯", "à®•à¯‡à®Ÿà¯"]
    MINUTE_WORDS = ["à®¨à®¿à®®à®¿à®Ÿà®®à¯", "à®¨à®¿à®®à®¿à®Ÿà®™à¯à®•à®³à¯", "à®¨à®¿à®®à®¿à®·à®®à¯"]
    SCALE_WORDS = {"à®•à¯‹à®Ÿà®¿": "crore", "à®•à¯‹à®Ÿà®¿à®¯à¯‡": "crore",
                   "à®²à®Ÿà¯à®šà®®à¯": "lakh", "à®²à®Ÿà¯à®šà®¤à¯à®¤à¯": "lakh", "à®‡à®²à®Ÿà¯à®šà®®à¯": "lakh",
                   "à®†à®¯à®¿à®°à®®à¯": "thousand", "à®†à®¯à®¿à®°à®¤à¯à®¤à¯": "thousand"}
    CURRENCY_WORDS = {"à®°à¯‚à®ªà®¾à®¯à¯": "rupee", "à®°à¯‚": "rupee", "à®°à¯à®ªà®¾à®¯à¯": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "à®®à¯à®¤à®²à®¾à®µà®¤à¯"}
    ORDINAL_SUFFIX = "à®†à®µà®¤à¯"
    ORDINAL_INPUT_SUFFIXES = ["à®†à®µà®¤à¯", "à®µà®¤à¯", "à®†à®®à¯", "à®®à¯"]

    # Consonants that double onto a preceding combining form (à®µà®²à¯à®²à®¿à®©à®®à¯).
    _DOUBLE = {"à®•": "à®•à¯", "à®š": "à®šà¯", "à®¤": "à®¤à¯", "à®ª": "à®ªà¯"}

    # ------------------------------------------------------------------
    # Dravidian composition: fused hundreds/thousands + sandhi joins.
    # ------------------------------------------------------------------
    def _sandhi_join(self, parts):
        """Join number parts; after a combining form ending -à®¤à¯à®¤à¯/-à®±à¯à®±à¯ a
        following à®•/à®š/à®¤/à®ª doubles onto it (à®†à®¯à®¿à®°à®¤à¯à®¤à¯à®¤à¯ à®¤à¯Šà®³à¯à®³à®¾à®¯à®¿à®°à®®à¯)."""
        out = [p for p in parts if p]
        for i in range(len(out) - 1):
            if out[i].endswith(("à®¤à¯à®¤à¯", "à®±à¯à®±à¯")) and out[i + 1][0] in self._DOUBLE:
                out[i] += self._DOUBLE[out[i + 1][0]]
        return " ".join(out)

    def hundreds(self, h, rest):
        if rest:
            return self._sandhi_join([self.HUNDREDS_COMB[h], self.two_digit(rest)])
        return self.HUNDREDS[h]

    def under_thousand(self, n):
        if n < 100:
            return self.two_digit(n)
        return self.hundreds(n // 100, n % 100)

    def _thousands_word(self, t, combining):
        """1 -> à®†à®¯à®¿à®°à®®à¯/à®†à®¯à®¿à®°à®¤à¯à®¤à¯; else fuse multiplier: final -à¯ drops before
        à®†- (à®¨à®¾à®©à¯à®•à¯+à®†à®¯à®¿à®°à®®à¯ -> à®¨à®¾à®©à¯à®•à®¾à®¯à®¿à®°à®®à¯, à®à®®à¯à®ªà®¤à¯à®¤à¯à®¨à®¾à®©à¯à®•à¯ -> à®à®®à¯à®ªà®¤à¯à®¤à¯à®¨à®¾à®©à¯à®•à®¾à®¯à®¿à®°à®®à¯)."""
        base = "à®†à®¯à®¿à®°à®¤à¯à®¤à¯" if combining else "à®†à®¯à®¿à®°à®®à¯"
        if t == 1:
            return base
        return self.ONES[t][:-1] + "à®¾" + base[1:]

    def cardinal(self, n):
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n == 0:
            return self.ONES[0]
        crore, rest = divmod(n, 10 ** 7)
        lakh, rest = divmod(rest, 10 ** 5)
        thousand, low = divmod(rest, 1000)
        parts = []
        if crore:
            mult = "à®’à®°à¯" if crore == 1 else self.cardinal(crore)
            parts.append(f"{mult} " + ("à®•à¯‹à®Ÿà®¿à®¯à¯‡" if (lakh or thousand or low) else "à®•à¯‹à®Ÿà®¿"))
        if lakh:
            mult = "à®’à®°à¯" if lakh == 1 else self.two_digit(lakh)
            parts.append(f"{mult} " + ("à®²à®Ÿà¯à®šà®¤à¯à®¤à¯" if (thousand or low) else "à®²à®Ÿà¯à®šà®®à¯"))
        if thousand:
            parts.append(self._thousands_word(thousand, bool(low)))
        if low:
            parts.append(self.under_thousand(low))
        return self._sandhi_join(parts)

    def year(self, y):
        """Tamil reads years as full cardinals:
        1947 -> à®†à®¯à®¿à®°à®¤à¯à®¤à¯à®¤à¯ à®¤à¯Šà®³à¯à®³à®¾à®¯à®¿à®°à®¤à¯à®¤à¯ à®¨à®¾à®±à¯à®ªà®¤à¯à®¤à¯‡à®´à¯,
        2026 -> à®‡à®°à®£à¯à®Ÿà®¾à®¯à®¿à®°à®¤à¯à®¤à¯ à®‡à®°à¯à®ªà®¤à¯à®¤à®¾à®±à¯."""
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # 12à®†à®µà®¤à¯/12à®µà®¤à¯ -> à®ªà®©à¯à®©à®¿à®°à®£à¯à®Ÿà®¾à®µà®¤à¯; 12à®†à®®à¯/12à®®à¯ -> à®ªà®©à¯à®©à®¿à®°à®£à¯à®Ÿà®¾à®®à¯.
        suffix = "à®†à®®à¯" if src_suffix in ("à®†à®®à¯", "à®®à¯") else "à®†à®µà®¤à¯"
        if n == 1:
            return "à®®à¯à®¤à®²à®¾à®®à¯" if suffix == "à®†à®®à¯" else "à®®à¯à®¤à®²à®¾à®µà®¤à¯"
        base = self.cardinal(n)
        if base[-1] in ("à¯", "à¯"):
            # drop final -u / virama and merge the suffix vowel: -à®¾à®µà®¤à¯ / -à®¾à®®à¯
            return base[:-1] + "à®¾" + suffix[1:]
        if base[-1] in ("à®¿", "à¯€", "à¯ˆ"):
            # glide insertion: à®•à¯‹à®Ÿà®¿ -> à®•à¯‹à®Ÿà®¿à®¯à®¾à®µà®¤à¯
            return base + "à®¯à®¾" + suffix[1:]
        return base + suffix

    # Tamil case markers attach with sandhi to noun-words
    _ATTACH_NOUNS = {"à®°à¯‚à®ªà®¾à®¯à¯", "à®Ÿà®¾à®²à®°à¯", "à®šà®¤à®µà¯€à®¤à®®à¯", "à®®à®£à®¿", "à®¨à®¿à®®à®¿à®Ÿà®®à¯",
                     "à®µà®¿à®¨à®¾à®Ÿà®¿", "à®•à®¿à®²à¯‹à®®à¯€à®Ÿà¯à®Ÿà®°à¯", "à®•à®¿à®²à¯‹", "à®®à¯€à®Ÿà¯à®Ÿà®°à¯", "à®²à®¿à®Ÿà¯à®Ÿà®°à¯",
                     "à®•à®¿à®°à®¾à®®à¯", "à®Ÿà®¿à®•à®¿à®°à®¿", "à®®à¯ˆà®²à¯"}

    def attach_postposition(self, rep, tok):
        t = tok.lstrip("-")
        last = rep.rsplit(" ", 1)[-1]
        base = rep[:len(rep)-len(last)]
        num_obl = {"à®’à®©à¯à®±à¯":"à®’à®©à¯à®±","à®‡à®°à®£à¯à®Ÿà¯":"à®‡à®°à®£à¯à®Ÿ","à®®à¯‚à®©à¯à®±à¯":"à®®à¯‚à®©à¯à®±",
                   "à®¨à®¾à®©à¯à®•à¯":"à®¨à®¾à®©à¯à®•","à®à®¨à¯à®¤à¯":"à®à®¨à¯à®¤","à®†à®±à¯":"à®†à®±","à®à®´à¯":"à®à®´",
                   "à®Žà®Ÿà¯à®Ÿà¯":"à®Žà®Ÿà¯à®Ÿ","à®’à®©à¯à®ªà®¤à¯":"à®’à®©à¯à®ªà®¤","à®ªà®¤à¯à®¤à¯":"à®ªà®¤à¯à®¤","à®ªà¯‚à®œà¯à®œà®¿à®¯à®®à¯":"à®ªà¯‚à®œà¯à®œà®¿à®¯à®¤à¯à®¤"}
        if last in num_obl:                       # number + case (hyphen markers)
            if t in ("à®‡à®²à¯","à®²à¯"): return base+num_obl[last]+"à®¿à®²à¯"
            if t in ("à®•à¯à®•à¯","à®•à¯"): return base+num_obl[last]+"à®¿à®±à¯à®•à¯"
            if t in ("à®‡à®²à®¿à®°à¯à®¨à¯à®¤à¯","à®²à¯à®‡à®°à¯à®¨à¯à®¤à¯"): return base+num_obl[last]+"à®¿à®²à®¿à®°à¯à®¨à¯à®¤à¯"
            return None
        if last not in self._ATTACH_NOUNS:
            return None
        if t in ("à®•à¯à®•à¯", "à®•à¯"):
            if last == "à®¨à®¿à®®à®¿à®Ÿà®®à¯":
                return base + "à®¨à®¿à®®à®¿à®Ÿà®¤à¯à®¤à®¿à®±à¯à®•à¯"
            if last == "à®šà®¤à®µà¯€à®¤à®®à¯":
                return base + "à®šà®¤à®µà¯€à®¤à®¤à¯à®¤à®¿à®±à¯à®•à¯"
            return base + last + "à®•à¯à®•à¯"
        if t in ("à®‡à®²à¯", "à®²à¯", "à®‡à®²à®¿à®°à¯à®¨à¯à®¤à¯", "à®²à¯à®‡à®°à¯à®¨à¯à®¤à¯"):
            if last.endswith("à®®à¯"):
                return base + last[:-2] + "à®¤à¯à®¤à®¿à®²à¯"
            return base + last + "à®‡à®²à¯"
        return None


class Telugu(LangBase):
    code = "te"
    name = "Telugu"
    native_digits = "à±¦à±§à±¨à±©à±ªà±«à±¬à±­à±®à±¯"

    ONES = [
        "à°¸à±à°¨à±à°¨à°¾", "à°’à°•à°Ÿà°¿", "à°°à±†à°‚à°¡à±", "à°®à±‚à°¡à±", "à°¨à°¾à°²à±à°—à±", "à°à°¦à±", "à°†à°°à±", "à°à°¡à±", "à°Žà°¨à°¿à°®à°¿à°¦à°¿", "à°¤à±Šà°®à±à°®à°¿à°¦à°¿",
        "à°ªà°¦à°¿", "à°ªà°¦à°•à±Šà°‚à°¡à±", "à°ªà°¨à±à°¨à±†à°‚à°¡à±", "à°ªà°¦à°®à±‚à°¡à±", "à°ªà°¦à±à°¨à°¾à°²à±à°—à±", "à°ªà°¦à°¿à°¹à±‡à°¨à±", "à°ªà°¦à°¹à°¾à°°à±", "à°ªà°¦à°¿à°¹à±‡à°¡à±", "à°ªà°¦à±à°¦à±†à°¨à°¿à°®à°¿à°¦à°¿", "à°ªà°‚à°¦à±Šà°®à±à°®à°¿à°¦à°¿",
        "à°‡à°°à°µà±ˆ", "à°‡à°°à°µà±ˆ à°’à°•à°Ÿà°¿", "à°‡à°°à°µà±ˆ à°°à±†à°‚à°¡à±", "à°‡à°°à°µà±ˆ à°®à±‚à°¡à±", "à°‡à°°à°µà±ˆ à°¨à°¾à°²à±à°—à±", "à°‡à°°à°µà±ˆ à°à°¦à±", "à°‡à°°à°µà±ˆ à°†à°°à±", "à°‡à°°à°µà±ˆ à°à°¡à±", "à°‡à°°à°µà±ˆ à°Žà°¨à°¿à°®à°¿à°¦à°¿", "à°‡à°°à°µà±ˆ à°¤à±Šà°®à±à°®à°¿à°¦à°¿",
        "à°®à±à°ªà±à°ªà±ˆ", "à°®à±à°ªà±à°ªà±ˆ à°’à°•à°Ÿà°¿", "à°®à±à°ªà±à°ªà±ˆ à°°à±†à°‚à°¡à±", "à°®à±à°ªà±à°ªà±ˆ à°®à±‚à°¡à±", "à°®à±à°ªà±à°ªà±ˆ à°¨à°¾à°²à±à°—à±", "à°®à±à°ªà±à°ªà±ˆ à°à°¦à±", "à°®à±à°ªà±à°ªà±ˆ à°†à°°à±", "à°®à±à°ªà±à°ªà±ˆ à°à°¡à±", "à°®à±à°ªà±à°ªà±ˆ à°Žà°¨à°¿à°®à°¿à°¦à°¿", "à°®à±à°ªà±à°ªà±ˆ à°¤à±Šà°®à±à°®à°¿à°¦à°¿",
        "à°¨à°²à°­à±ˆ", "à°¨à°²à°­à±ˆ à°’à°•à°Ÿà°¿", "à°¨à°²à°­à±ˆ à°°à±†à°‚à°¡à±", "à°¨à°²à°­à±ˆ à°®à±‚à°¡à±", "à°¨à°²à°­à±ˆ à°¨à°¾à°²à±à°—à±", "à°¨à°²à°­à±ˆ à°à°¦à±", "à°¨à°²à°­à±ˆ à°†à°°à±", "à°¨à°²à°­à±ˆ à°à°¡à±", "à°¨à°²à°­à±ˆ à°Žà°¨à°¿à°®à°¿à°¦à°¿", "à°¨à°²à°­à±ˆ à°¤à±Šà°®à±à°®à°¿à°¦à°¿",
        "à°¯à°¾à°­à±ˆ", "à°¯à°¾à°­à±ˆ à°’à°•à°Ÿà°¿", "à°¯à°¾à°­à±ˆ à°°à±†à°‚à°¡à±", "à°¯à°¾à°­à±ˆ à°®à±‚à°¡à±", "à°¯à°¾à°­à±ˆ à°¨à°¾à°²à±à°—à±", "à°¯à°¾à°­à±ˆ à°à°¦à±", "à°¯à°¾à°­à±ˆ à°†à°°à±", "à°¯à°¾à°­à±ˆ à°à°¡à±", "à°¯à°¾à°­à±ˆ à°Žà°¨à°¿à°®à°¿à°¦à°¿", "à°¯à°¾à°­à±ˆ à°¤à±Šà°®à±à°®à°¿à°¦à°¿",
        "à°…à°°à°µà±ˆ", "à°…à°°à°µà±ˆ à°’à°•à°Ÿà°¿", "à°…à°°à°µà±ˆ à°°à±†à°‚à°¡à±", "à°…à°°à°µà±ˆ à°®à±‚à°¡à±", "à°…à°°à°µà±ˆ à°¨à°¾à°²à±à°—à±", "à°…à°°à°µà±ˆ à°à°¦à±", "à°…à°°à°µà±ˆ à°†à°°à±", "à°…à°°à°µà±ˆ à°à°¡à±", "à°…à°°à°µà±ˆ à°Žà°¨à°¿à°®à°¿à°¦à°¿", "à°…à°°à°µà±ˆ à°¤à±Šà°®à±à°®à°¿à°¦à°¿",
        "à°¡à±†à°¬à±à°¬à±ˆ", "à°¡à±†à°¬à±à°¬à±ˆ à°’à°•à°Ÿà°¿", "à°¡à±†à°¬à±à°¬à±ˆ à°°à±†à°‚à°¡à±", "à°¡à±†à°¬à±à°¬à±ˆ à°®à±‚à°¡à±", "à°¡à±†à°¬à±à°¬à±ˆ à°¨à°¾à°²à±à°—à±", "à°¡à±†à°¬à±à°¬à±ˆ à°à°¦à±", "à°¡à±†à°¬à±à°¬à±ˆ à°†à°°à±", "à°¡à±†à°¬à±à°¬à±ˆ à°à°¡à±", "à°¡à±†à°¬à±à°¬à±ˆ à°Žà°¨à°¿à°®à°¿à°¦à°¿", "à°¡à±†à°¬à±à°¬à±ˆ à°¤à±Šà°®à±à°®à°¿à°¦à°¿",
        "à°Žà°¨à°­à±ˆ", "à°Žà°¨à°­à±ˆ à°’à°•à°Ÿà°¿", "à°Žà°¨à°­à±ˆ à°°à±†à°‚à°¡à±", "à°Žà°¨à°­à±ˆ à°®à±‚à°¡à±", "à°Žà°¨à°­à±ˆ à°¨à°¾à°²à±à°—à±", "à°Žà°¨à°­à±ˆ à°à°¦à±", "à°Žà°¨à°­à±ˆ à°†à°°à±", "à°Žà°¨à°­à±ˆ à°à°¡à±", "à°Žà°¨à°­à±ˆ à°Žà°¨à°¿à°®à°¿à°¦à°¿", "à°Žà°¨à°­à±ˆ à°¤à±Šà°®à±à°®à°¿à°¦à°¿",
        "à°¤à±Šà°‚à°­à±ˆ", "à°¤à±Šà°‚à°­à±ˆ à°’à°•à°Ÿà°¿", "à°¤à±Šà°‚à°­à±ˆ à°°à±†à°‚à°¡à±", "à°¤à±Šà°‚à°­à±ˆ à°®à±‚à°¡à±", "à°¤à±Šà°‚à°­à±ˆ à°¨à°¾à°²à±à°—à±", "à°¤à±Šà°‚à°­à±ˆ à°à°¦à±", "à°¤à±Šà°‚à°­à±ˆ à°†à°°à±", "à°¤à±Šà°‚à°­à±ˆ à°à°¡à±", "à°¤à±Šà°‚à°­à±ˆ à°Žà°¨à°¿à°®à°¿à°¦à°¿", "à°¤à±Šà°‚à°­à±ˆ à°¤à±Šà°®à±à°®à°¿à°¦à°¿",
    ]
    # Indo-Aryan style fields kept for interface completeness; the fused
    # forms below (à°µà°‚à°¦/à°¨à±‚à°Ÿ/à°µà°‚à°¦à°²à±/à°µà°‚à°¦à°² etc.) are what cardinal() actually uses.
    HUNDRED = "à°µà°‚à°¦"
    THOUSAND = "à°µà±†à°¯à±à°¯à°¿"
    LAKH = "à°²à°•à±à°·"
    CRORE = "à°•à±‹à°Ÿà°¿"

    LETTERS = {
        "A": "à°", "B": "à°¬à±€", "C": "à°¸à±€", "D": "à°¡à±€", "E": "à°ˆ", "F": "à°Žà°«à±",
        "G": "à°œà±€", "H": "à°¹à±†à°šà±", "I": "à°", "J": "à°œà±‡", "K": "à°•à±‡", "L": "à°Žà°²à±",
        "M": "à°Žà°‚", "N": "à°Žà°¨à±", "O": "à°“", "P": "à°ªà±€", "Q": "à°•à±à°¯à±‚", "R": "à°†à°°à±",
        "S": "à°Žà°¸à±", "T": "à°Ÿà±€", "U": "à°¯à±‚", "V": "à°µà±€", "W": "à°¡à°¬à±à°²à±à°¯à±‚",
        "X": "à°Žà°•à±à°¸à±", "Y": "à°µà±ˆ", "Z": "à°œà±†à°¡à±",
    }

    MONTHS = ["à°œà°¨à°µà°°à°¿", "à°«à°¿à°¬à±à°°à°µà°°à°¿", "à°®à°¾à°°à±à°šà°¿", "à°à°ªà±à°°à°¿à°²à±", "à°®à±‡", "à°œà±‚à°¨à±",
              "à°œà±‚à°²à±ˆ", "à°†à°—à°¸à±à°Ÿà±", "à°¸à±†à°ªà±à°Ÿà±†à°‚à°¬à°°à±", "à°…à°•à±à°Ÿà±‹à°¬à°°à±", "à°¨à°µà°‚à°¬à°°à±", "à°¡à°¿à°¸à±†à°‚à°¬à°°à±"]
    MONTH_ALIASES = {"à°¸à±†à°ªà±à°Ÿà±†à°‚à°¬à°°à±": 9, "à°…à°•à±à°Ÿà±‹à°¬à°°à±": 10, "à°¨à°µà°‚à°¬à°°à±": 11, "à°¡à°¿à°¸à±†à°‚à°¬à°°à±": 12}

    SYM = {
        "rupee": "à°°à±‚à°ªà°¾à°¯à°²à±", "dollar": "à°¡à°¾à°²à°°à±à°²à±", "euro": "à°¯à±‚à°°à±‹à°²à±",
        "pound": "à°ªà±Œà°‚à°¡à±à°²à±", "cent": "à°¸à±†à°‚à°Ÿà±à°²à±",
        "percent": "à°¶à°¾à°¤à°‚", "paisa": "à°ªà±ˆà°¸à°²à±",
        "point": "à°ªà°¾à°¯à°¿à°‚à°Ÿà±",
        "minus": "à°®à±ˆà°¨à°¸à±", "plus": "à°ªà±à°²à°¸à±", "equals": "à°¸à°®à°¾à°¨à°‚",
        "times": "à°‡à°‚à°Ÿà±‚", "divide": "à°­à°¾à°—à°¿à°‚à°šà°—à°¾",
        "squared": "à°µà°°à±à°—à°‚", "sqrt": "à°µà°°à±à°—à°®à±‚à°²à°‚",
        "lt": "à°•à°‚à°Ÿà±‡ à°¤à°•à±à°•à±à°µ", "gt": "à°•à°‚à°Ÿà±‡ à°Žà°•à±à°•à±à°µ",
        "plus_minus": "à°ªà±à°²à°¸à± à°®à±ˆà°¨à°¸à±",
        "at": "à°Žà°Ÿà± à°¦ à°°à±‡à°Ÿà±", "underscore": "à°…à°‚à°¡à°°à± à°¸à±à°•à±‹à°°à±", "dot": "à°¡à°¾à°Ÿà±",
        "slash": "à°¸à±à°²à°¾à°·à±", "hash": "à°¹à±à°¯à°¾à°·à±", "amp": "à°®à°°à°¿à°¯à±",
        "half": "à°…à°°", "quarter": "à°ªà°¾à°µà±", "three_quarters": "à°®à±à°ªà±à°ªà°¾à°µà±",
        "degree": "à°¡à°¿à°—à±à°°à±€à°²à±", "celsius": "à°¡à°¿à°—à±à°°à±€à°² à°¸à±†à°²à±à°¸à°¿à°¯à°¸à±", "fahrenheit": "à°¡à°¿à°—à±à°°à±€à°² à°«à°¾à°°à±†à°¨à±â€Œà°¹à±€à°Ÿà±",
        "minute": "à°¨à°¿à°®à°¿à°·à°¾à°²à±", "per": "à°ªà°°à±", "second": "à°¸à±†à°•à°¨à±à°²à±",
    }

    TIME_ON_HOUR = "{h} à°—à°‚à°Ÿà°²à±"
    TIME_HM = "{h} à°—à°‚à°Ÿà°² {m} à°¨à°¿à°®à°¿à°·à°¾à°²à±"
    RANGE = "{a} à°¨à±à°‚à°šà°¿ {b}"
    RANGE_TO = "{a} à°¨à±à°‚à°šà°¿ {b} à°µà°°à°•à±"
    RATIO = "{a} à°¨à°¿à°·à±à°ªà°¤à±à°¤à°¿ {b}"
    POWER = "{base} à°ªà°µà°°à± {exp}"
    FRACTION = "{num} à°¬à±ˆ {den}"
    DECADE_SHORT = "{tens}à°²"
    DECADE_FULL = "{tens}à°² à°¦à°¶à°•à°‚"

    PERIODS = {"morning": "à°‰à°¦à°¯à°‚", "afternoon": "à°®à°§à±à°¯à°¾à°¹à±à°¨à°‚",
               "evening": "à°¸à°¾à°¯à°‚à°¤à±à°°à°‚", "night": "à°°à°¾à°¤à±à°°à°¿"}

    TIME_MARKERS = ["à°—à°‚à°Ÿà°²à°•à±", "à°—à°‚à°Ÿà°²à°•à°¿", "à°—à°‚à°Ÿà°•à±", "à°—à°‚à°Ÿà°•à°¿", "à°—à°‚à°Ÿà°²", "à°—à°‚à°Ÿà°²à±", "à°—à°‚"]
    MARKER_SUFFIXES = {"à°—à°‚à°Ÿà°²à°•à±": "à°•à±", "à°—à°‚à°Ÿà°²à°•à°¿": "à°•à°¿", "à°—à°‚à°Ÿà°•à±": "à°•à±",
                       "à°—à°‚à°Ÿà°•à°¿": "à°•à°¿", "à°¨à°¿à°®à°¿à°·à°¾à°²à°•à±": "à°•à±", "à°¨à°¿à°®à°¿à°·à°¾à°¨à°¿à°•à°¿": "à°•à°¿"}
    PERIOD_DETECT = {"à°‰à°¦à°¯à°‚": "morning", "à°ªà±Šà°¦à±à°¦à±à°¨": "morning", "à°ªà±Šà°¦à±à°¦à±à°¨à±à°¨": "morning",
                     "à°¤à±†à°²à±à°²à°µà°¾à°°à±à°œà°¾à°®à±à°¨": "morning", "à°®à°§à±à°¯à°¾à°¹à±à°¨à°‚": "afternoon",
                     "à°¸à°¾à°¯à°‚à°¤à±à°°à°‚": "evening", "à°°à°¾à°¤à±à°°à°¿": "night"}
    YEAR_WORDS = ["à°¸à°‚à°µà°¤à±à°¸à°°à°‚", "à°¸à°‚à°µà°¤à±à°¸à°°à°‚à°²à±‹", "à°à°¡à°¾à°¦à°¿", "à°¸à°‚"]
    ID_KEYWORDS = ["à°¨à°‚à°¬à°°à±", "à°¨à°‚à°¬à°°à±", "à°¨à±†à°‚à°¬à°°à±", "à°¨à±†à°‚à°¬à°°à±", "à°–à°¾à°¤à°¾", "à°«à±‹à°¨à±", "à°®à±Šà°¬à±ˆà°²à±", "à°•à±‹à°¡à±", "à°†à°°à±à°¡à°°à±", "à°Žà°•à±à°¸à±à°ªà±à°°à±†à°¸à±", "express", "à°…à°•à±Œà°‚à°Ÿà±", "à°Ÿà°¿à°•à±†à°Ÿà±", "à°°à°¿à°œà°¿à°¸à±à°Ÿà±à°°à±‡à°·à°¨à±", "à°¯à±‚à°œà°°à±", "à°µà°¾à°¯à°¿à°¦à°¾", "à°…à°‚à°•à±†", "à°²à°¾à°µà°¾à°¦à±‡à°µà±€", "à°ªà°¿à°¨à±"]
    ID_AFTER_WORDS = ["à°Žà°‚à°¡à±", "à°®à±à°—à°¿à°¸à±‡", "à°®à±à°—à±à°¸à±à°¤à±à°‚à°¦à°¿", "à°šà°¿à°µà°°", "à°šà°¿à°µà°°à°¿", "à°…à°‚à°•à±†à°²à°¤à±‹", "à°¡à°¯à°²à±", "à°ªà°¿à°¨à±", "à°ªà°¿à°¨à±â€Œà°•à±‹à°¡à±", "pin", "pincode", "à°•à°¾à°²à±", "à°¸à°‚à°ªà±à°°à°¦à°¿à°‚à°šà°‚à°¡à°¿"]
    LABEL_WORDS = ["à°—à°¦à°¿", "à°µà°¾à°°à±à°¡à±", "à°…à°§à°¿à°•à°°à°£", "à°…à°§à±à°¯à°¾à°¯à°‚", "à°¤à°°à°—à°¤à°¿", "à°­à°¾à°—à°‚", "à°¸à±†à°•à±à°·à°¨à±",
                   "à°ªà±à°²à°¾à°Ÿà±à°«à°¾à°°à°®à±", "à°ªà±à°²à°¾à°Ÿà±â€Œà°«à°¾à°°à°®à±", "à°¸à±€à°Ÿà±", "à°¸à±€à°Ÿà±", "à°•à±‹à°šà±", "à°—à±‡à°Ÿà±",
                   "à°°à±‚à°®à±", "à°¹à°¾à°²à±", "à°Ÿà±†à°°à±à°®à°¿à°¨à°²à±", "à°•à±Œà°‚à°Ÿà°°à±"]
    MINUTE_WORDS = ["à°¨à°¿à°®à°¿à°·à°¾à°²à°•à±", "à°¨à°¿à°®à°¿à°·à°¾à°¨à°¿à°•à°¿", "à°¨à°¿à°®à°¿à°·à°¾à°²à±", "à°¨à°¿à°®à°¿à°·à°¾à°²", "à°¨à°¿à°®à°¿à°·à°‚"]
    SCALE_WORDS = {"à°•à±‹à°Ÿà°¿": "crore", "à°•à±‹à°Ÿà±€": "crore", "à°•à±‹à°Ÿà±à°²à±": "crore", "à°•à±‹à°Ÿà±à°²": "crore",
                   "à°²à°•à±à°·": "lakh", "à°²à°•à±à°·à°¾": "lakh", "à°²à°•à±à°·à°²à±": "lakh", "à°²à°•à±à°·à°²": "lakh",
                   "à°µà±†à°¯à±à°¯à°¿": "thousand", "à°µà±‡à°²à±": "thousand", "à°µà±‡à°²": "thousand"}
    CURRENCY_WORDS = {"à°°à±‚à°ªà°¾à°¯à°²à±": "rupee", "à°°à±‚à°ªà°¾à°¯à°²": "rupee", "à°°à±‚à°ªà°¾à°¯à°¿": "rupee",
                      "à°°à±‚à°ªà°¾à°¯": "rupee", "à°°à±‚": "rupee"}

    ORDINAL_EXCEPTIONS = {1: "à°®à±Šà°¦à°Ÿà°¿"}
    ORDINAL_SUFFIX = "à°µ"
    ORDINAL_INPUT_SUFFIXES = ["à°µ", "à°µà°¦à°¿", "à±‹"]

    # ==================================================================
    # Dravidian composition â€” fused/sandhi forms (overrides base).
    # ==================================================================

    def hundreds(self, h, rest):
        """100 à°µà°‚à°¦ / à°¨à±‚à°Ÿ; 200+ à°µà°‚à°¦à°²à± / combining à°µà°‚à°¦à°²."""
        if h == 1:
            if rest:
                return f"à°¨à±‚à°Ÿ {self.ONES[rest]}"
            return "à°µà°‚à°¦"
        if rest:
            return f"{self.ONES[h]} à°µà°‚à°¦à°² {self.ONES[rest]}"
        return f"{self.ONES[h]} à°µà°‚à°¦à°²à±"

    def under_thousand(self, n):
        if n < 100:
            return self.ONES[n]
        return self.hundreds(n // 100, n % 100)

    def _thousands(self, t, has_rest):
        """t in 1..99. 1000 à°µà±†à°¯à±à°¯à°¿ (same before a remainder); 2+ à°µà±‡à°²à±/à°µà±‡à°²."""
        if t == 1:
            return "à°µà±†à°¯à±à°¯à°¿"
        return f"{self.ONES[t]} {'à°µà±‡à°²' if has_rest else 'à°µà±‡à°²à±'}"

    def _lakhs(self, l, has_rest):
        """l in 1..99. à°²à°•à±à°· / sandhi à°²à°•à±à°·à°¾; 2+ à°²à°•à±à°·à°²à±/à°²à°•à±à°·à°²."""
        if l == 1:
            return "à°²à°•à±à°·à°¾" if has_rest else "à°²à°•à±à°·"
        return f"{self.ONES[l]} {'à°²à°•à±à°·à°²' if has_rest else 'à°²à°•à±à°·à°²à±'}"

    def _crores(self, c, has_rest):
        """à°•à±‹à°Ÿà°¿ / sandhi à°•à±‹à°Ÿà±€; 2+ à°•à±‹à°Ÿà±à°²à±/à°•à±‹à°Ÿà±à°²."""
        if c == 1:
            return "à°•à±‹à°Ÿà±€" if has_rest else "à°•à±‹à°Ÿà°¿"
        return f"{self.cardinal(c)} {'à°•à±‹à°Ÿà±à°²' if has_rest else 'à°•à±‹à°Ÿà±à°²à±'}"

    def cardinal(self, n):
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n < 100:
            return self.ONES[n]
        if n < 1000:
            return self.under_thousand(n)
        crore, rest = divmod(n, 10**7)
        lakh, rest = divmod(rest, 10**5)
        thousand, sub = divmod(rest, 1000)
        parts = []
        if crore:
            parts.append(self._crores(crore, bool(lakh or thousand or sub)))
        if lakh:
            parts.append(self._lakhs(lakh, bool(thousand or sub)))
        if thousand:
            parts.append(self._thousands(thousand, bool(sub)))
        if sub:
            parts.append(self.under_thousand(sub))
        return " ".join(parts)

    def year(self, y):
        """Spoken Telugu reads 1100-1999 hundred-style â€” 1947 à°ªà°‚à°¦à±Šà°®à±à°®à°¿à°¦à°¿
        à°µà°‚à°¦à°² à°¨à°²à°­à±ˆ à°à°¡à± (the "à°µà±†à°¯à±à°¯à°¿ à°¤à±Šà°®à±à°®à°¿à°¦à°¿ à°µà°‚à°¦à°²â€¦" full cardinal sounds
        stiff; flagged by native review). 2000+ stays full cardinal:
        2026 à°°à±†à°‚à°¡à± à°µà±‡à°² à°‡à°°à°µà±ˆ à°†à°°à±."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            if lo:
                return f"{self.ONES[hi]} à°µà°‚à°¦à°² {self.ONES[lo]}"
            return f"{self.ONES[hi]} à°µà°‚à°¦à°²à±"
        return self.cardinal(y)

    def _ordinal_compose(self, n, src_suffix):
        # Formal -à°µ ordinal with the regular Telugu stem change:
        # à°°à±†à°‚à°¡à± -> à°°à±†à°‚à°¡à°µ, à°Žà°¨à°¿à°®à°¿à°¦à°¿ -> à°Žà°¨à°¿à°®à°¿à°¦à°µ, à°‡à°°à°µà±ˆ -> à°‡à°°à°µà°¯à±à°¯à°µ, à°µà°‚à°¦ -> à°µà°‚à°¦à°µ.
        if n in self.ORDINAL_EXCEPTIONS and not src_suffix:
            return self.ORDINAL_EXCEPTIONS[n]
        word = self.cardinal(n)
        if word.endswith("à±") or word.endswith("à°¿"):   # à± / à°¿ matra
            stem = word[:-1]
        elif word.endswith("à±ˆ"):                            # à±ˆ matra
            stem = word[:-1] + "à°¯à±à°¯"
        else:
            stem = word
        return f"{stem}{self.ORDINAL_SUFFIX}"

    # ------------------------------------------------------------------
    # Inherited Telugu grammar hooks. Upstream cited an external review, not
    # independently verified human/native-speaker approval.
    # ------------------------------------------------------------------

    # metric units read as full words, not letter names
    UNIT_WORDS = {
        "KM": "à°•à°¿à°²à±‹à°®à±€à°Ÿà°°à±à°²à±", "KG": "à°•à°¿à°²à±‹à°²à±", "GM": "à°—à±à°°à°¾à°®à±à°²à±",
        "MG": "à°®à°¿à°²à±à°²à±€à°—à±à°°à°¾à°®à±à°²à±", "MCG": "à°®à±ˆà°•à±à°°à±‹à°—à±à°°à°¾à°®à±à°²à±",
        "ML": "à°®à°¿à°²à±à°²à±€à°²à±€à°Ÿà°°à±à°²à±", "CM": "à°¸à±†à°‚à°Ÿà±€à°®à±€à°Ÿà°°à±à°²à±", "MM": "à°®à°¿à°²à±à°²à±€à°®à±€à°Ÿà°°à±à°²à±",
        "H": "à°…à°µà°°à±", "HR": "à°…à°µà°°à±",
    }

    NUMBER_WORD = "à°¨à°‚à°¬à°°à±"
    ABBREV_UNITS = {"MINS": "à°¨à°¿à°®à°¿à°·à°¾à°²à±", "MIN": "à°¨à°¿à°®à°¿à°·à°¾à°²à±",
                    "HRS": "à°—à°‚à°Ÿà°²à±", "HR": "à°—à°‚à°Ÿ",
                    "SECS": "à°¸à±†à°•à°¨à±à°²à±", "SEC": "à°¸à±†à°•à°¨à±à°²à±", "KMS": "à°•à°¿à°²à±‹à°®à±€à°Ÿà°°à±à°²à±"}

    # singular noun forms used after "1" (à°’à°• à°°à±‚à°ªà°¾à°¯à°¿, à°’à°• à°—à°‚à°Ÿ)
    _NOUN_SINGULAR = {"à°°à±‚à°ªà°¾à°¯à°²à±": "à°°à±‚à°ªà°¾à°¯à°¿", "à°¡à°¾à°²à°°à±à°²à±": "à°¡à°¾à°²à°°à±",
                      "à°—à°‚à°Ÿà°²à±": "à°—à°‚à°Ÿ", "à°¨à°¿à°®à°¿à°·à°¾à°²à±": "à°¨à°¿à°®à°¿à°·à°‚", "à°ªà±ˆà°¸à°²à±": "à°ªà±ˆà°¸à°¾", "à°¸à±†à°•à°¨à±à°²à±": "à°¸à±†à°•à°¨à±"}

    def _oblique_last(self, words):
        """Oblique stem for the final word before an appended noun:
        à°µà°‚à°¦à°²à±/à°µà±‡à°²à±/à°²à°•à±à°·à°²à±/à°•à±‹à°Ÿà±à°²à± -> à°µà°‚à°¦à°²/à°µà±‡à°²/à°²à°•à±à°·à°²/à°•à±‹à°Ÿà±à°² ; à°’à°•à°Ÿà°¿ -> à°’à°•."""
        last = words.rsplit(" ", 1)[-1]
        if last.endswith("à°²à±"):
            return words[:-1]          # strip à± : à°µà°‚à°¦à°²à± -> à°µà°‚à°¦à°²
        return words

    def decimal(self, int_part, frac_digits):
        frac = " ".join(self.digit(int(d)) for d in frac_digits)
        return f"{self._oblique_last(self.cardinal(int_part))} " \
               f"{self.SYM['point']} {frac}"

    def join_amount(self, words, noun):
        if words == "à°’à°•à°Ÿà°¿":
            return f"à°’à°• {self._NOUN_SINGULAR.get(noun, noun)}"
        return f"{self._oblique_last(words)} {noun}"

    def time_phrase(self, h, m, period_key=None):
        h12 = h % 12 or 12
        hw = "à°’à°•" if h12 == 1 else self.two_digit(h12)
        hnoun = "à°—à°‚à°Ÿ" if h12 == 1 else ("à°—à°‚à°Ÿà°²à±" if m == 0 else "à°—à°‚à°Ÿà°²")
        if h12 == 1 and m != 0:
            hnoun = "à°—à°‚à°Ÿ"
        if m == 0:
            core = f"{hw} {hnoun}"
        elif m == 1:
            core = f"{hw} {hnoun} à°’à°• à°¨à°¿à°®à°¿à°·à°‚"
        else:
            core = f"{hw} {hnoun} {self.two_digit(m)} à°¨à°¿à°®à°¿à°·à°¾à°²à±"
        if period_key:
            return f"{self.PERIODS[period_key]} {core}"
        return core

    # postpositions appearing as a separate token right after a rendered
    # span get attached with correct sandhi ("à°—à°‚à°Ÿà°²à± à°•à°¿" -> "à°—à°‚à°Ÿà°²à°•à±")
    _PP_SPACED = {"à°µà°°à°•à±", "à°¦à°¾à°•à°¾", "à°²à±‹à°ªà±", "à°¨à±à°‚à°¡à°¿", "à°¨à±à°‚à°šà°¿",
                  "à°ªà°¾à°Ÿà±", "à°•à±à°°à°¿à°¤à°‚", "à°¤à°°à±à°µà°¾à°¤", "à°®à±à°‚à°¦à±", "à°šà±Šà°ªà±à°ªà±à°¨", "à°•à°‚à°Ÿà±‡", "à°¦à°—à±à°—à°°", "à°²à±‹à°ªà°²"}
    _PP_GLUE = {"à°•à°¿", "à°•à±", "à°¨à°¿", "à°¨", "à°¨à±‡", "à°•à±‡", "à°²à±‹", "à°¤à±‹",
                "à°²à±‹à°¨à°¿", "à°²à±‹à°•à°¿", "à°²à±‹à°¨à±‡", "à°•à°²à±à°²à°¾"}
    # à°—à°¾ keeps the nominative: à°°à±‚à°ªà°¾à°¯à°²à± + à°—à°¾ -> à°°à±‚à°ªà°¾à°¯à°²à±à°—à°¾
    _PP_NOMINATIVE = {"à°—à°¾", "à°—à°¾à°¨à±‡"}
    # common counted nouns: a plural-final cardinal right before these takes
    # the oblique stem (à°®à±‚à°¡à± à°²à°•à±à°·à°²à± à°°à±‚à°ªà°¾à°¯à°²à± -> à°®à±‚à°¡à± à°²à°•à±à°·à°² à°°à±‚à°ªà°¾à°¯à°²à±)
    _COUNT_NOUNS = {"à°°à±‚à°ªà°¾à°¯à°²à±", "à°®à°‚à°¦à°¿", "à°•à°¿à°²à±‹à°²à±", "à°•à°¿à°²à±‹à°®à±€à°Ÿà°°à±à°²à±", "à°—à°‚à°Ÿà°²à±",
                    "à°¨à°¿à°®à°¿à°·à°¾à°²à±", "à°°à±‹à°œà±à°²à±", "à°¸à°‚à°µà°¤à±à°¸à°°à°¾à°²à±", "à°à°³à±à°²à±", "à°¸à°¾à°°à±à°²à±",
                    "à°®à°¾à°°à±à°•à±à°²à±", "à°“à°Ÿà±à°²à±", "à°¸à±€à°Ÿà±à°²à±", "à°ªà°°à±à°—à±à°²à±", "à°µà°¿à°•à±†à°Ÿà±à°²à±",
                    "à°²à±€à°Ÿà°°à±à°²à±", "à°—à±à°°à°¾à°®à±à°²à±", "à°Ÿà°¨à±à°¨à±à°²à±", "à°Žà°•à°°à°¾à°²à±", "à°¶à°¾à°¤à°‚",
                    "à°•à±‹à°Ÿà±à°²à±", "à°²à°•à±à°·à°²à±", "à°µà°¡à±à°¡à±€", "à°œà±€à°¤à°‚", "à°–à°°à±à°šà±",
                    "à°§à°°", "à°†à°¦à°¾à°¯à°‚", "à°²à°¾à°­à°‚", "à°¨à°·à±à°Ÿà°‚", "à°ªà±†à°Ÿà±à°Ÿà±à°¬à°¡à°¿", "à°°à±à°£à°‚",
                    "à°œà°°à°¿à°®à°¾à°¨à°¾", "à°¬à°¡à±à°œà±†à°Ÿà±", "à°®à±‹à°¤à°¾à°¦à±", "à°®à°¾à°¤à±à°°", "à°®à°¾à°¤à±à°°à°²à±", "à°Ÿà°¾à°¬à±à°²à±†à°Ÿà±", "à°•à±‹à°Ÿà±à°²", "à°²à°•à±à°·à°²",
                    "à°¦à±‚à°°à°‚", "à°¬à°°à±à°µà±", "à°µà±‡à°—à°‚", "à°®à±ˆà°³à±à°²à±", "à°®à±ˆà°³à±à°³à±", "à°…à°¡à±à°—à±à°²à±", "à°¸à±†à°•à°¨à±à°²à±",
                    "à°ªà±‡à°œà±€à°²à±", "à°¸à°®à°¯à°‚", "à°¸à°®à°¯à°¾à°¨à°¿à°•à°¿"}

    def scale_noun(self, canon, numstr):
        one = numstr.replace(",", "") == "1"
        return {"crore": "à°•à±‹à°Ÿà°¿" if one else "à°•à±‹à°Ÿà±à°²à±",
                "lakh": "à°²à°•à±à°·" if one else "à°²à°•à±à°·à°²à±",
                "thousand": "à°µà±†à°¯à±à°¯à°¿" if one else "à°µà±‡à°²à±"}.get(canon, canon)

    _PP_ROMAN = {"ki": "à°•à°¿", "ku": "à°•à±", "ni": "à°¨à°¿", "lo": "à°²à±‹",
                 "ga": "à°—à°¾", "ke": "à°•à±‡", "varaku": "à°µà°°à°•à±", "nunchi": "à°¨à±à°‚à°šà°¿",
                 "nundi": "à°¨à±à°‚à°¡à°¿", "daggara": "à°¦à°—à±à°—à°°"}

    def attach_postposition(self, rep, tok):
        tok = self._PP_ROMAN.get(tok.lower(), tok)
        last = rep.rsplit(" ", 1)[-1]
        if rep == "à°’à°•à°Ÿà°¿" and tok[:1].isascii() and tok[:1].isalpha():
            return "à°’à°• " + tok               # 1 litre -> à°’à°• litre
        if not last or not ("à°€" <= last[-1] <= "à±¿"):
            return None            # last word not Telugu (Latin unit etc.)
        if last in self.LETTERS.values():
            return None            # bare letter names take no glued case
        if tok == last:
            return rep             # duplicate word (à°µà°°à°•à± à°µà°°à°•à±) â€” consume
        plural = last.endswith("à°²à±")
        if tok in self._PP_NOMINATIVE:
            return rep + tok                 # à°°à±‚à°ªà°¾à°¯à°²à± à°—à°¾ -> à°°à±‚à°ªà°¾à°¯à°²à±à°—à°¾
        base_tok = tok.replace("\u200c", "")
        for end in ("à°¨à±", "à°¨à°¿", "à°•à±", "à°•à°¿", "à°²à±‹", "à°—à°¾", "à°¤à±‹", "à°ªà±ˆ"):
            if base_tok.endswith(end) and base_tok[:-len(end)] in self._COUNT_NOUNS:
                base_tok = base_tok[:-len(end)]
                break
        if plural and base_tok in self._COUNT_NOUNS:
            return rep[:-1] + " " + tok
        if rep == "à°’à°•à°Ÿà°¿" and tok in self._COUNT_NOUNS:
            return "à°’à°• " + self._NOUN_SINGULAR.get(tok, tok)
        if tok in self._PP_SPACED:
            return (rep[:-1] if plural else rep) + " " + tok
        if tok not in self._PP_GLUE:
            return None
        if last == "à°¶à°¾à°¤à°‚" and tok in ("à°•à°¿", "à°•à±"):
            return rep[:-1] + "à°¾à°¨à°¿à°•à°¿"        # à°¶à°¾à°¤à°‚ à°•à°¿ -> à°¶à°¾à°¤à°¾à°¨à°¿à°•à°¿
        if plural:
            if tok in ("à°•à°¿", "à°•à±"):
                return rep[:-1] + tok        # keep the source marker: à°—à°‚à°Ÿà°²à± à°•à°¿ -> à°—à°‚à°Ÿà°²à°•à°¿
            if tok == "à°¨à°¿":
                return rep[:-1] + "à°¨à±"       # à°°à±‚à°ªà°¾à°¯à°²à± à°¨à°¿ -> à°°à±‚à°ªà°¾à°¯à°²à°¨à±
            return rep[:-1] + tok            # à°—à°‚à°Ÿà°²à± à°²à±‹ -> à°—à°‚à°Ÿà°²à°²à±‹
        if tok in ("à°•à°¿", "à°•à±"):
            return rep + tok                 # keep the source marker: à°à°¦à± à°•à°¿ -> à°à°¦à±à°•à°¿
        return rep + tok


class Assamese(Bengali):
    """Assamese lexicon; Bengali grammar/entries remain the documented fallback.

    Number forms: AI4Bharat data; Gregorian months: Unicode CLDR as.xml.
    No new inference dependency and no translation of customer prose.
    """
    code = "as"
    name = "Assamese"
    if __package__:
        from .assamese_numbers import ONES, HUNDREDS
    else:
        from assamese_numbers import ONES, HUNDREDS
    HUNDRED = "à¦¶"
    THOUSAND = "à¦¹à¦¾à¦œà¦¾à§°"
    MONTHS = ['à¦œà¦¾à¦¨à§à§±à¦¾à§°à§€', 'à¦«à§‡à¦¬à§à§°à§à§±à¦¾à§°à§€', 'à¦®à¦¾à§°à§à¦š', 'à¦à¦ªà§à§°à¦¿à¦²', 'à¦®à§‡â€™', 'à¦œà§à¦¨', 'à¦œà§à¦²à¦¾à¦‡', 'à¦†à¦—à¦·à§à¦Ÿ', 'à¦›à§‡à¦ªà§à¦¤à§‡à¦®à§à¦¬à§°', 'à¦…à¦•à§à¦Ÿà§‹à¦¬à§°', 'à¦¨à§±à§‡à¦®à§à¦¬à§°', 'à¦¡à¦¿à¦šà§‡à¦®à§à¦¬à§°']
    SYM = {**Bengali.SYM, "rupee": "à¦Ÿà¦•à¦¾", "paisa": "à¦ªà¦‡à¦šà¦¾", "dollar": "à¦¡à¦²à¦¾à§°",
           "euro": "à¦‡à¦‰à§°à§‹", "pound": "à¦ªà¦¾à¦‰à¦£à§à¦¡"}
    CURRENCY_WORDS = {**Bengali.CURRENCY_WORDS, "à¦Ÿà¦•à¦¾": "rupee", "à¦Ÿà¦•à¦¾à§°": "rupee",
                      "à¦Ÿà¦•à¦¾à¦¤": "rupee", "à§°à§à¦ªà§€": "rupee"}
    SCALE_WORDS = {**Bengali.SCALE_WORDS, "à¦¹à¦¾à¦œà¦¾à§°": "thousand"}
    ID_KEYWORDS = Bengali.ID_KEYWORDS + ["à¦¨à¦®à§à¦¬à§°", "à¦à¦•à¦¾à¦‰à¦£à§à¦Ÿ", "à¦•à¦¾à§°à§à¦¡", "à¦¨à¦®à§à¦¬à§°à¦¤", "à¦…à¦‚à¦•"]
    NUMBER_WORD = "à¦¨à¦®à§à¦¬à§°"
    YEAR_WORDS = Bengali.YEAR_WORDS + ["à¦šà¦¨", "à¦šà¦¨à¦¤", "à¦¬à¦›à§°", "à¦¬à¦›à§°à¦¤"]
    TIME_HM = "{h} à¦¬à¦¾à¦œà¦¿ {m} à¦®à¦¿à¦¨à¦¿à¦Ÿ"
    RANGE = "{a}à§° à¦ªà§°à¦¾ {b}"
    RANGE_TO = "{a}à§° à¦ªà§°à¦¾ {b}à¦²à§ˆ"
    FRACTION = "{num} à¦¬à¦¾à¦‡ {den}"
    SINGULAR_NOUNS = {}

    def hundreds(self, h, rest):
        return self.HUNDREDS[h] + (" " + self.two_digit(rest) if rest else "")

    def year(self, y):
        return self.cardinal(y)

LANGUAGES = {cls.code: cls() for cls in (
    Hindi, Bengali, Assamese, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu
)}
'''),
    'mithi_tts.text_norm.tn.normalize': (False, r'''"""Dependency-free, reviewable TTS normalization.

Indian English and ten Indic language tables share one corrected parser.
The selected locale controls numbers, currency, dates, units and identifiers;
it does not translate prose or choose/change the model conditioning tag.
Hindi speech output folds nukta and ends standalone à¤¹à¥ˆ with a danda outside
explicit question sentences (ending in ?).
These are intentional pronunciation/pause workarounds, not spelling corrections.
Invalid/ambiguous tokens fall back to digit and separator words.
FR3_TN_STRICT=1 retains the strict audit contract.
See docs/NORMALIZATION_REVIEW.md for the contract and remaining listening gates.
"""
# -*- coding: utf-8 -*-
# The former generator/indic_norm package is absent from this repository.
# This tracked file is the source of truth; protect edits with tests/test_normalize.py.
import re
import os
import json
import sys
import argparse
from datetime import date
import unicodedata
from functools import lru_cache

# Keep CLI execution and the imported table module on the same LangBase class.
if __name__ == "__main__":
    sys.modules.setdefault("normalize", sys.modules[__name__])


LANGUAGE_NAMES = {"en-IN": "Indian English", "hi": "Hindi", "te": "Telugu",
    "ta": "Tamil", "bn": "Bengali", "as": "Assamese", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "mr": "Marathi", "or": "Odia", "pa": "Punjabi"}
_LANGUAGE_ALIASES = {alias.lower(): code for code, name in LANGUAGE_NAMES.items()
    for alias in (code, name, code if code == "en-IN" else code + "-IN")}
_LANGUAGE_ALIASES.update({"en": "en-IN", "english": "en-IN", "en-us": "en-IN",
    "en-gb": "en-IN", "indian-english": "en-IN", "oriya": "or", "od": "or"})

# Input aliases, not translations of arbitrary prose. Keep identifiers digitwise
# even when callers write OTP in native script (including Indian English mode).
_OTP_INPUT_WORDS = {"à¤“à¤Ÿà¥€à¤ªà¥€", "à°“à°Ÿà±€à°ªà±€", "à°“à°Ÿà°¿à°ªà°¿", "à®“à®Ÿà®¿à®ªà®¿", "à®“à®Ÿà¯€à®ªà®¿", "à¦“à¦Ÿà¦¿à¦ªà¦¿",
                    "àª“àªŸà«€àªªà«€", "à²’à²Ÿà²¿à²ªà²¿", "à²“à²Ÿà²¿à²ªà²¿", "à´’à´Ÿà´¿à´ªà´¿", "à´“à´Ÿà´¿à´ªà´¿", "à¬“à¬Ÿà¬¿à¬ªà¬¿", "à¨“à¨Ÿà©€à¨ªà©€"}
_PAISA_INPUT_WORDS = {"paise", "paisa", "à¤ªà¥ˆà¤¸à¤¾", "à¤ªà¥ˆà¤¸à¥‡", "à°ªà±ˆà°¸à°¾", "à°ªà±ˆà°¸à°²à±", "à®ªà¯ˆà®šà®¾",
                      "à¦ªà¦¯à¦¼à¦¸à¦¾", "àªªà«ˆàª¸àª¾", "à²ªà³ˆà²¸à³†", "à´ªàµˆà´¸", "à¬ªà¬‡à¬¸à¬¾", "à¨ªà©ˆà¨¸à¨¾", "à¨ªà©ˆà¨¸à©‡"}


def resolve_language(language):
    if not isinstance(language, str) or language.lower() not in _LANGUAGE_ALIASES:
        raise NormalizationError("unsupported_normalization_language", "Choose one of the documented normalization languages.")
    return _LANGUAGE_ALIASES[language.lower()]


@lru_cache(maxsize=12)
def language_table(language):
    code = resolve_language(language)
    if code == "en-IN":
        return LANG
    if __package__:
        from .indic_languages import LANGUAGES
    else:
        from indic_languages import LANGUAGES
    return LANGUAGES[code]


class NormalizationError(ValueError):
    """A safe-to-return error code. Never includes the caller's financial text."""
    def __init__(self, code, message, span=None, source=None):
        self.span, self.source = span, source
        self.code = code
        super().__init__(message)


_TAMIL_MONTHS = ["à®œà®©à®µà®°à®¿", "à®ªà®¿à®ªà¯à®°à®µà®°à®¿", "à®®à®¾à®°à¯à®šà¯", "à®à®ªà¯à®°à®²à¯", "à®®à¯‡", "à®œà¯‚à®©à¯",
                 "à®œà¯‚à®²à¯ˆ", "à®†à®•à®¸à¯à®Ÿà¯", "à®šà¯†à®ªà¯à®Ÿà®®à¯à®ªà®°à¯", "à®…à®•à¯à®Ÿà¯‹à®ªà®°à¯", "à®¨à®µà®®à¯à®ªà®°à¯", "à®Ÿà®¿à®šà®®à¯à®ªà®°à¯"]
# Calendar days only. Native-speaker/audio review is required before rollout.
_TAMIL_ORDINALS = ["", "à®®à¯à®¤à®²à®¾à®®à¯", "à®‡à®°à®£à¯à®Ÿà®¾à®®à¯", "à®®à¯‚à®©à¯à®±à®¾à®®à¯", "à®¨à®¾à®©à¯à®•à®¾à®®à¯", "à®à®¨à¯à®¤à®¾à®®à¯",
    "à®†à®±à®¾à®®à¯", "à®à®´à®¾à®®à¯", "à®Žà®Ÿà¯à®Ÿà®¾à®®à¯", "à®’à®©à¯à®ªà®¤à®¾à®®à¯", "à®ªà®¤à¯à®¤à®¾à®®à¯", "à®ªà®¤à®¿à®©à¯Šà®©à¯à®±à®¾à®®à¯",
    "à®ªà®©à¯à®©à®¿à®°à®£à¯à®Ÿà®¾à®®à¯", "à®ªà®¤à®¿à®©à¯à®®à¯‚à®©à¯à®±à®¾à®®à¯", "à®ªà®¤à®¿à®©à®¾à®©à¯à®•à®¾à®®à¯", "à®ªà®¤à®¿à®©à¯ˆà®¨à¯à®¤à®¾à®®à¯", "à®ªà®¤à®¿à®©à®¾à®±à®¾à®®à¯",
    "à®ªà®¤à®¿à®©à¯‡à®´à®¾à®®à¯", "à®ªà®¤à®¿à®©à¯†à®Ÿà¯à®Ÿà®¾à®®à¯", "à®ªà®¤à¯à®¤à¯Šà®©à¯à®ªà®¤à®¾à®®à¯", "à®‡à®°à¯à®ªà®¤à®¾à®®à¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®±à®¾à®®à¯",
    "à®‡à®°à¯à®ªà®¤à¯à®¤à®¿à®°à®£à¯à®Ÿà®¾à®®à¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à¯à®®à¯‚à®©à¯à®±à®¾à®®à¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à¯à®¨à®¾à®©à¯à®•à®¾à®®à¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à¯ˆà®¨à¯à®¤à®¾à®®à¯",
    "à®‡à®°à¯à®ªà®¤à¯à®¤à®¾à®±à®¾à®®à¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à¯‡à®´à®¾à®®à¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à¯†à®Ÿà¯à®Ÿà®¾à®®à¯", "à®‡à®°à¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®ªà®¤à®¾à®®à¯",
    "à®®à¯à®ªà¯à®ªà®¤à®¾à®®à¯", "à®®à¯à®ªà¯à®ªà®¤à¯à®¤à¯Šà®©à¯à®±à®¾à®®à¯"]


def _ascii_digits(text):
    # str.isdigit also accepts superscripts. Only Unicode DECIMAL digits belong
    # here; keep math superscripts for the math matcher.
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)


# Hindi speech compatibility, not a general Unicode diacritic remover. Apply to
# FINAL output because number/date/letter/URL expansions can introduce nukta.
# Keep every other script, accent, vowel sign and nasal mark untouched.
_HINDI_NUKTA_FOLD = str.maketrans({
    "\u093c": None,
    "\u0929": "\u0928", "\u0931": "\u0930", "\u0934": "\u0933",
    "\u0958": "\u0915", "\u0959": "\u0916", "\u095a": "\u0917",
    "\u095b": "\u091c", "\u095c": "\u0921", "\u095d": "\u0922",
    "\u095e": "\u092b", "\u095f": "\u092f",
})


def _calendar_check(day, month, year=2000, span=None):
    # A missing year can include Feb 29; never invent a year in the output.
    try:
        date(year, month, day)
    except ValueError:
        raise NormalizationError("invalid_date", "Invalid calendar date; correct the input.", span=span) from None


def _sentence_before(text, start):
    return re.split(r"[.!?;à¥¤à¥¥\n]", text[max(0, start - 100):start])[-1]


def _sentence_after(text, end):
    return re.split(r"[.!?;à¥¤à¥¥\n]", text[end:end + 100])[0]


def _valid_grouping(number):
    integer = number.lstrip("+-âˆ’").split(".")[0]
    return "," not in integer or bool(re.fullmatch(
        r"[0-9]{1,3}(?:,[0-9]{3})+|[0-9]{1,2}(?:,[0-9]{2})*,[0-9]{3}", integer))


def _month_lookup(locale):
    return {**_ENGLISH_MONTHS, **{name.lower(): i + 1 for i, name in enumerate(locale.MONTHS)},
            **{name.lower(): i for name, i in locale.MONTH_ALIASES.items()}}


def validate_text(text, *, date_order="DMY", two_digit_year_base=None, language="en-IN"):
    """Validate before destructive cleanup. Dates default to explicit DMY policy.

    Two-digit years require a caller-supplied century (1900 or 2000). Short dates
    in date context must be spelled out; ordinary fractions remain fractions.
    Plain text only: SSML/templates are rejected, not stripped and spoken.
    """
    if not isinstance(text, str):
        raise NormalizationError("invalid_text", "text must be a string.")
    if len(text) > 20000:
        raise NormalizationError("text_too_long", "text exceeds the 20000 character limit.")
    if date_order not in ("DMY", "MDY"):
        raise NormalizationError("invalid_date_order", "date_order must be DMY or MDY.")
    if two_digit_year_base is not None and (type(two_digit_year_base) is not int or two_digit_year_base not in (1900, 2000)):
        raise NormalizationError("invalid_century", "two_digit_year_base must be 1900 or 2000.")
    locale = language_table(language)
    t = _ascii_digits(text)
    if (bad := re.search(r"<\s*(?:/?[A-Za-z_]|[!?])|[{}]|\$\{", t)):
        raise NormalizationError("markup_or_placeholder", "Resolve templates and send plain text, not markup/SSML.", span=bad.span())
    for i, c in enumerate(t):
        if (unicodedata.category(c) == "Cc" and c not in "\n\r\t") or unicodedata.category(c) == "Co":
            raise NormalizationError("control_character", "Unsupported control character in text.", span=(i, i + 1))
    # URI numeric components are identifiers, not dates, amounts or clocks.
    t = re.sub(r"https?://\S+|www\.\S+|\b[\w.%+\-]+@[\w.\-]+\.[A-Za-z]{2,}", lambda m: " " * len(m[0]), t)
    if (bad := re.search(r"[0-9]{31,}", t)):
        raise NormalizationError("number_too_long", "Numeric spans are limited to 30 digits.", span=bad.span())
    if (bad := re.search(r"\d{4}-\d{2}-\d{2}T\d", t, re.I)):
        raise NormalizationError("unsupported_timestamp", "Spell out the date, time and timezone separately.", span=bad.span())
    for m in re.finditer(r"(?<![\w.])[+-]?[0-9][0-9,]*(?:\.[0-9]+)?", t):
        if len(re.sub(r"\D", "", m[0])) > 30:
            raise NormalizationError("number_too_long", "Numeric spans are limited to 30 digits.", span=m.span())
        if not _valid_grouping(m[0].rstrip(",")):
            raise NormalizationError("invalid_grouping", "Invalid number grouping; send ungrouped digits or standard grouping.", span=m.span())
    for m in re.finditer(r"(?<![\w\d])([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})(?!\d)", t):
        _calendar_check(int(m[3]), int(m[2]), int(m[1]), span=m.span())
    for m in re.finditer(r"(?<![\w\d-])([0-9]{1,2})([/.-])([0-9]{1,2})\2([0-9]{2,4})(?!\d)", t):
        y = int(m[4])
        if len(m[4]) != 4:
            if len(m[4]) != 2 or two_digit_year_base is None:
                raise NormalizationError("ambiguous_year", "Use a four-digit year or explicitly configure the century.", span=m.span())
            y += two_digit_year_base
        a, b = int(m[1]), int(m[3])
        _calendar_check(a if date_order == "DMY" else b, b if date_order == "DMY" else a, y, span=m.span())
    months = {**{name: i + 1 for i, name in enumerate(_TAMIL_MONTHS)}, **_month_lookup(locale)}
    month_re = "|".join(sorted(map(re.escape, months), key=len, reverse=True))
    ordinal_re = "|".join(sorted(map(re.escape, set(locale.ORDINAL_INPUT_SUFFIXES) | {"st", "nd", "rd", "th", "à®†à®®à¯", "à®®à¯"}), key=len, reverse=True))
    for pattern, reverse in [
        (rf"(?<!\w)([0-9]{{1,2}})(?:-?(?:{ordinal_re}))?[ -]+({month_re})(?!\w)(?:[ ,\-]+([0-9]{{4}}))?", False),
        (rf"(?<!\w)({month_re})[ -]+([0-9]{{1,2}})(?:-?(?:{ordinal_re}))?(?!\d)(?:,?\s+([0-9]{{4}}))?", True),
    ]:
        for m in re.finditer(pattern, t, re.I):
            day, month = (m[2], m[1]) if reverse else (m[1], m[2])
            _calendar_check(int(day), months[month.lower()], int(m[3]) if m[3] else 2000, span=m.span())
    tamil_month_re = "|".join(_TAMIL_MONTHS)
    for m in re.finditer(rf"({tamil_month_re})\s+(\d{{1,2}})(?:-?(?:à®†à®®à¯|à®®à¯))?(?:\s*à®¤à¯‡à®¤à®¿)?(?:\s+(\d{{4}}))?", t):
        _calendar_check(int(m[2]), months[m[1]], int(m[3]) if m[3] else 2000, span=m.span())
    if (bad := re.search(r"\b(?:on|by|before|after|date|due)\s+[0-9]{1,2}/[0-9]{1,2}(?![/\d])", t, re.I)):
        raise NormalizationError("ambiguous_short_date", "Spell out the month and year for a short date.", span=bad.span())
    if (bad := re.search(r"à®¤à¯‡à®¤à®¿\s+[0-9]{1,2}\.\s*[0-9]{4}", t)):
        raise NormalizationError("conflicting_date", "Conflicting date components; confirm the intended date.", span=bad.span())
    rupee_words = [word for word, key in locale.CURRENCY_WORDS.items() if key == "rupee"]
    rupee_aliases = "|".join(sorted(map(re.escape, rupee_words + ["INR", "Rs"]), key=len, reverse=True))
    rupee_prefix = rf"(?:â‚¹|(?<!\w)(?:{rupee_aliases})\.?)"
    minor_units = "|".join(sorted(map(re.escape, _PAISA_INPUT_WORDS), key=len, reverse=True))
    if (bad := re.search(rf"{rupee_prefix}\s*[+-]?[0-9][0-9,.]*\s*(?:{minor_units})(?=$|[\s.,!?;à¥¤à¥¥-])", t, re.I)):
        raise NormalizationError("conflicting_currency_unit", "Rupee prefix conflicts with a paise suffix; confirm the unit.", span=bad.span())
    if (bad := re.search(rf"(?:{rupee_prefix}|[$â‚¬Â£]|\b(?:USD|EUR|GBP))\s*[+-]?[0-9][0-9,.]*\s*%", t, re.I)):
        raise NormalizationError("conflicting_currency_unit", "A value cannot have both a currency prefix and percent suffix.", span=bad.span())
    for m in re.finditer(r"(?<![\w:])([0-9]{1,3}):([0-9]{2}):([0-9]{2})(?![\d:])", t):
        if int(m[2]) > 59 or int(m[3]) > 59:
            raise NormalizationError("invalid_duration", "Duration minutes and seconds must be below 60.", span=m.span())
    no_durations = re.sub(r"[0-9]+:[0-9]{2}:[0-9]{2}", lambda m: " " * len(m[0]), t)
    for m in re.finditer(r"(?<![\w:])([0-9]{1,2})(?:[:.]([0-9]{2}))?\s*(a\.?m\.?|p\.?m\.?)(?!\w)|(?<![\w:])([0-9]{1,2}):([0-9]{2})(?![\d:])", no_durations, re.I):
        if m[1]:
            valid = 1 <= int(m[1]) <= 12 and int(m[2] or 0) < 60
        else:
            # Explicit ratio context is not a clock.
            if re.search(r"\b(?:ratio|odds|proportion)\b", _sentence_before(t, m.start()), re.I):
                continue
            valid = int(m[4]) < 24 and int(m[5]) < 60
        if not valid:
            raise NormalizationError("invalid_time", "Invalid clock time; use 00:00-23:59 or 1-12 with am/pm.", span=m.span())

class LangBase:
    # ---- identity -------------------------------------------------------
    code = ""            # ISO 639-1: hi bn ta te mr gu kn ml pa or
    name = ""
    native_digits = ""   # the 10 native digit glyphs "à¥¦à¥§à¥¨à¥©à¥ªà¥«à¥¬à¥­à¥®à¥¯" etc.

    # ---- numbers --------------------------------------------------------
    ONES = []            # words for 0..99 (100 entries, native script)
    HUNDRED = ""         # word used in "X hundred" composition (Indo-Aryan)
    THOUSAND = ""
    LAKH = ""
    CRORE = ""

    # ---- letters (English letter names written in native script) --------
    LETTERS = {}         # {"A": "...", ..., "Z": "..."}

    # ---- calendar -------------------------------------------------------
    MONTHS = []          # 12 native month names (Jan..Dec, Gregorian)
    MONTH_ALIASES = {}   # accepted alternate input spelling -> month 1..12

    # ---- symbols / unit words -------------------------------------------
    # Values are native-script spoken words. Keys are fixed; the engine
    # relies on every key existing.
    SYM = {
        "rupee": "", "dollar": "", "euro": "", "pound": "", "cent": "",
        "percent": "",          # % symbol
        "point": "",            # decimal point word (Hindi: à¤¦à¤¶à¤®à¤²à¤µ)
        "minus": "", "plus": "", "equals": "", "times": "", "divide": "",
        "squared": "",          # Â² (Hindi: à¤•à¤¾ à¤µà¤°à¥à¤— â€” appended after operand)
        "sqrt": "",             # âˆš (spoken before operand)
        "lt": "", "gt": "",     # < >
        "plus_minus": "",       # Â±
        "at": "",               # @  (Hindi: à¤à¤Ÿ à¤¦ à¤°à¥‡à¤Ÿ)
        "underscore": "",
        "dot": "",              # . in emails/URLs
        "slash": "",            # / in URLs
        "hash": "",             # #
        "amp": "",              # &
        "half": "", "quarter": "", "three_quarters": "",
        "degree": "", "celsius": "", "fahrenheit": "",
        "minute": "",           # time minutes word
    }

    # ---- templates (use {placeholders}) ----------------------------------
    TIME_ON_HOUR = ""    # "{h} à¤¬à¤œà¥‡"
    TIME_HM = ""         # "{h} à¤¬à¤œà¤•à¤° {m} à¤®à¤¿à¤¨à¤Ÿ"
    RANGE = ""           # "{a} à¤¸à¥‡ {b}"          (10â€“20)
    RANGE_TO = ""        # "{a} à¤¸à¥‡ {b} à¤¤à¤•"       (time ranges)
    RATIO = ""           # "{a} à¤…à¤¨à¥à¤ªà¤¾à¤¤ {b}"
    POWER = ""           # "{base} à¤•à¥€ à¤˜à¤¾à¤¤ {exp}"
    FRACTION = ""        # "{num} à¤¬à¤Ÿà¤¾ {den}"     (2/7)
    DECADE_SHORT = ""    # "{tens} à¤•à¥‡"           (90s)
    DECADE_FULL = ""     # "{tens} à¤•à¤¾ à¤¦à¤¶à¤•"       (1980s)

    # Period words emitted before the time (order: period first, then time).
    PERIODS = {"morning": "", "afternoon": "", "evening": "", "night": ""}

    # ---- source-text detection vocab (native words the engine must
    #      recognise in the INPUT). Include all common variants/spellings.
    TIME_MARKERS = []      # e.g. ["à¤¬à¤œà¥‡", "à¤¬à¤œà¤•à¤°"]
    # markers that carry a case suffix which must be re-attached to the
    # rendered time ("10 à°—à°‚à°Ÿà°²à°•à±" -> "à°ªà°¦à°¿ à°—à°‚à°Ÿà°²à°•à±", not "à°ªà°¦à°¿ à°—à°‚à°Ÿà°²à±")
    MARKER_SUFFIXES = {}   # e.g. {"à°—à°‚à°Ÿà°²à°•à±": "à°•à±"}
    PERIOD_DETECT = {}     # native word -> "morning"/"afternoon"/"evening"/"night"
    YEAR_WORDS = []        # e.g. ["à¤¸à¤¾à¤²", "à¤µà¤°à¥à¤·", "à¤¸à¤¨à¥"]
    ID_KEYWORDS = []       # native keywords marking identifiers: à¤¨à¤‚à¤¬à¤°, à¤–à¤¾à¤¤à¤¾ ...
    ID_AFTER_WORDS = []    # words AFTER a digit-run marking it as an
                           # identifier ("7104 à°¤à±‹ à°Žà°‚à°¡à± à°…à°¯à±à°¯à±‡ number")
    LABEL_WORDS = []       # native label words read as cardinal: à¤•à¤®à¤°à¤¾, à¤…à¤¨à¥à¤šà¥à¤›à¥‡à¤¦ ...
    MINUTE_WORDS = []      # native "minute" words in input, e.g. ["à¤®à¤¿à¤¨à¤Ÿ"]
    SCALE_WORDS = {}       # native scale word in input -> "crore"/"lakh"/"thousand"
    CURRENCY_WORDS = {}    # native currency words in input -> canonical key
                           # e.g. {"à¤°à¥à¤ªà¤¯à¥‡": "rupee", "à¤°à¥": "rupee"}

    # ---- ordinals ---------------------------------------------------------
    ORDINAL_EXCEPTIONS = {}   # n -> word (Hindi: {1: "à¤ªà¤¹à¤²à¤¾", 2: "à¤¦à¥‚à¤¸à¤°à¤¾", ...})
    ORDINAL_SUFFIX = ""       # default suffix appended to cardinal (Hindi: à¤µà¤¾à¤)
    # Suffixes seen in input that should be preserved on output when possible
    ORDINAL_INPUT_SUFFIXES = []   # e.g. ["à¤µà¥€à¤‚", "à¤µà¤¾à¤", "à¤µà¥‡à¤‚", "à¤µà¤¾à¤‚"]

    # ======================================================================
    # Composition methods â€” Indo-Aryan defaults; Dravidian modules override.
    # ======================================================================

    def digit(self, d: int) -> str:
        return self.ONES[d]

    def two_digit(self, n: int) -> str:
        return self.ONES[n]

    def hundreds(self, h: int, rest: int) -> str:
        """h in 1..9, rest 0..99 -> 'X sau [rest]'."""
        out = f"{self.ONES[h]} {self.HUNDRED}"
        if rest:
            out += f" {self.two_digit(rest)}"
        return out

    def under_thousand(self, n: int) -> str:
        if n < 100:
            return self.two_digit(n)
        return self.hundreds(n // 100, n % 100)

    def cardinal(self, n: int) -> str:
        """Indian-system cardinal. Override in Dravidian modules."""
        if n < 0:
            return f"{self.SYM['minus']} {self.cardinal(-n)}"
        if n < 1000:
            return self.under_thousand(n)
        parts = []
        crore, n = divmod(n, 10**7)
        lakh, n = divmod(n, 10**5)
        thousand, n = divmod(n, 1000)
        if crore:
            parts.append(f"{self.cardinal(crore)} {self.CRORE}")
        if lakh:
            parts.append(f"{self.two_digit(lakh)} {self.LAKH}")
        if thousand:
            parts.append(f"{self.two_digit(thousand)} {self.THOUSAND}")
        if n:
            parts.append(self.under_thousand(n))
        return " ".join(parts)

    def year(self, y: int) -> str:
        """Year-style reading. Indo-Aryan: 1947 -> 'unnis sau saintalis'.
        Dravidian modules usually override to full cardinal."""
        if 1100 <= y <= 1999:
            hi, lo = divmod(y, 100)
            out = f"{self.two_digit(hi)} {self.HUNDRED}"
            if lo:
                out += f" {self.two_digit(lo)}"
            return out
        return self.cardinal(y)

    def letter(self, c: str) -> str:
        return self.LETTERS.get(c.upper(), c)

    def month(self, i: int) -> str:
        """i is 1..12"""
        return self.MONTHS[i - 1]

    def time_phrase(self, h: int, m: int, period_key=None) -> str:
        # _resolve_time may deliberately hand back a 24-hour hour (languages
        # that transcribe am/pm keep "14:30" as "fourteen thirty"); only fold
        # into 12-hour when it has not already been decided upstream.
        if self.AMPM_WORDS and h > 12:
            h12 = h
        else:
            h12 = h % 12
            if h12 == 0:
                h12 = 12
        if m == 0:
            core = self.TIME_ON_HOUR.format(h=self.two_digit(h12))
        else:
            core = self.TIME_HM.format(h=self.two_digit(h12), m=self.two_digit(m))
        if period_key:
            return f"{self.PERIODS[period_key]} {core}"
        return core

    def ordinal(self, n: int, src_suffix: str = "") -> str:
        if n in self.ORDINAL_EXCEPTIONS and not src_suffix:
            return self.ORDINAL_EXCEPTIONS[n]
        return self._ordinal_compose(n, src_suffix)

    def _ordinal_compose(self, n: int, src_suffix: str) -> str:
        """Default: cardinal + suffix. Languages with stem changes override."""
        base = self.ORDINAL_EXCEPTIONS.get(n)
        if base and not src_suffix:
            return base
        suffix = src_suffix if src_suffix in self.ORDINAL_INPUT_SUFFIXES else self.ORDINAL_SUFFIX
        return f"{self.cardinal(n)}{suffix}"

    def decimal(self, int_part: int, frac_digits: str) -> str:
        frac = " ".join(self.digit(int(d)) for d in frac_digits)
        return f"{self.cardinal(int_part)} {self.SYM['point']} {frac}"

    def decade(self, tens: int, full: bool) -> str:
        """tens: e.g. 90 for '90s', 80 for '1980s'."""
        tpl = self.DECADE_FULL if full else self.DECADE_SHORT
        return tpl.format(tens=self.two_digit(tens))

    # ------------------------------------------------------------------
    # Grammar hooks â€” no-ops by default; languages with case/sandhi
    # morphology (Telugu, ...) override these.
    # ------------------------------------------------------------------

    # unit initialisms that read as full native words instead of letters
    # (e.g. Telugu KM -> à°•à°¿à°²à±‹à°®à±€à°Ÿà°°à±à°²à±)
    UNIT_WORDS = {}

    # written abbreviations that are not speakable words â€” expanded to native
    # words wherever they appear (mins -> à°¨à°¿à°®à°¿à°·à°¾à°²à±). Keys uppercased.
    ABBREV_UNITS = {}

    # spoken word for the "No." / "no." abbreviation before digits
    NUMBER_WORD = ""

    # When non-empty the language transcribes the am/pm token literally and the
    # engine stops INFERRING a period from the hour. Indo-Aryan languages say
    # the period naturally ("shaam paanch baje"), so they leave this empty;
    # English must not have "in the evening" inserted into text that only said
    # "pm". Keys are lowercase, dots stripped.
    AMPM_WORDS = {}
    COUNT_ONE = None
    SINGULAR_NOUNS = {}

    def join_amount(self, words: str, noun: str) -> str:
        """Join number words with a following noun we insert (currency,
        scale, unit). Languages may apply oblique/singular morphology."""
        if words == self.ONES[1]:
            return f"{self.COUNT_ONE or words} {self.SINGULAR_NOUNS.get(noun, noun)}"
        return f"{words} {noun}"

    def scale_noun(self, canon: str, numstr: str) -> str:
        """Scale word used in 'â‚¹3 crore'-type phrases; languages may
        pluralize when the number is not 1."""
        return {"crore": self.CRORE, "lakh": self.LAKH,
                "thousand": self.THOUSAND}.get(canon, canon)

    def attach_postposition(self, rep: str, tok: str):
        """Called when a native token `tok` immediately follows a rendered
        span `rep`. Return the merged string (consuming the token), or None
        to leave both untouched."""
        return None



# ----------------------------------------------------------------------------
# Static tables
# ----------------------------------------------------------------------------

# All Indic decimal digit blocks -> ASCII
_INDIC_DIGIT_BLOCKS = [
    0x0966,  # Devanagari
    0x09E6,  # Bengali
    0x0A66,  # Gurmukhi
    0x0AE6,  # Gujarati
    0x0B66,  # Odia
    0x0BE6,  # Tamil
    0x0C66,  # Telugu
    0x0CE6,  # Kannada
    0x0D66,  # Malayalam
]
_DIGIT_TRANS = {}
for _base in _INDIC_DIGIT_BLOCKS:
    for _i in range(10):
        _DIGIT_TRANS[_base + _i] = ord("0") + _i

# Emoji / decorative ranges to strip entirely
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # symbols & pictographs, emoticons, transport...
    "\U00002700-\U000027BF"   # dingbats
    "\U00002600-\U000026FF"   # misc symbols
    "\U0001F1E6-\U0001F1FF"   # flags
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0001F000-\U0001F0FF"
    "\U00002B00-\U00002BFF"   # arrows/stars (â­)
    "\U00002300-\U000023FF"   # misc technical (âŒ› â³ â¬)
    "\U00002190-\U000021FF"   # arrows (â†’)
    "\U000025A0-\U000025FF"   # geometric (â— â–¶)
    "\U00002022"              # bullet â€¢
    "\U00002605\U00002606"    # â˜… â˜†
    "\U00002122\U000000A9\U000000AE"  # â„¢ Â© Â®
    # NOTE: ZWNJ/ZWJ (200C/200D) are deliberately NOT stripped â€” they are
    # meaningful orthography in Telugu/Malayalam (à°†à°¨à±â€Œà°²à±ˆà°¨à±). Only the truly
    # invisible ZWSP/BOM are removed, without inserting a space.
    "`~|\\\\{}\\[\\]Â«Â»â€¹â€ºâ€žâ€šâ€¢Â·Â¤Â§â€ â€¡"
    "]+"
)
_ZW_STRIP = {0x200B: None, 0x200E: None, 0x200F: None, 0xFEFF: None}

_ENGLISH_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Established acronyms that stay in Latin script (never spelled out unless
# they are part of an alphanumeric identifier).
_ACRONYM_WHITELIST = {
    "UPI", "EMI", "API", "MRI", "GST", "SEBI", "MCX", "IGI", "HDFC", "ISRO",
    "IMD", "SBI", "RBI", "ICICI", "IRCTC", "ATM", "SIM", "OTP", "PIN", "CEO",
    "CFO", "CTO", "IPL", "BCCI", "DNA", "RNA", "WHO", "NASA", "PDF", "GPS",
    "LED", "LCD", "USB", "WIFI", "AI", "IT", "TV", "FM", "AC", "EV", "CBI",
    "NIA", "ED", "GDP", "CPI", "FII", "DII", "IPO", "NSE", "BSE", "KYC",
    "NEFT", "RTGS", "IMPS", "IFSC", "CVV", "PAN", "UID", "NRI", "MLA", "MP",
    "PM", "CM", "DM", "SP", "IAS", "IPS", "NEET", "JEE", "UPSC", "CBSE",
    "ICSE", "BA", "MA", "MBA", "MBBS", "AIIMS", "ISKCON", "DLF", "TCS",
    "LTIMINDTREE", "ONGC", "NTPC", "BHEL", "SAIL", "GAIL", "LIC",
    "RAM", "ROM", "CPU", "GPU", "SSD", "HDD", "GPA", "SUV", "SLA", "GDP",
    "UIDAI", "FASTAG", "GSTIN", "OK", "US", "UK", "UAE", "USA", "EU", "UN",
    "PSU", "FIR", "RTI", "PIL", "GPRS", "HD", "UHD", "CCTV", "AQI", "UV",
}

# Units read letter-by-letter (initialisms). Keys uppercased for lookup.
_UNIT_INITIALISMS = {
    "GB", "MB", "KB", "TB", "PB", "MAH", "MPH", "KMPH", "MHZ", "GHZ", "KHZ",
    "HZ", "KWH", "RPM", "HP", "BHP", "CC", "DPI", "PPI", "FPS", "MBPS",
    "GBPS", "KBPS", "PSI", "BPM", "V", "KV", "W", "KW", "MW", "MA", "AH",
    "WH", "DB", "PPM", "ML", "MG", "KG", "KM", "CM", "MM", "GM", "MCG", "IU",
}

# English "big number" scale words that stay in Latin (5 million -> à¤ªà¤¾à¤à¤š million)
_EN_SCALES = {"million", "billion", "trillion", "mn", "bn", "millions",
              "billions"}

_ID_KEYWORDS_EN = {
    "phone", "mobile", "card", "account", "otp", "pin", "code", "order",
    "ref", "reference", "train", "flight", "ticket", "tracking", "license",
    "licence", "booking", "pnr", "upi", "imei", "registration", "number",
    "no", "ending", "khasra", "plot", "docket", "complaint", "case", "loan",
    "policy", "invoice", "receipt", "transaction", "txn", "utr", "id",
    "extension", "helpline", "hotline", "whatsapp", "aadhaar", "aadhar",
    "passport", "cvv", "gstin",
}

_LABEL_WORDS_EN = {
    "hall", "window", "counter", "room", "ward", "terminal", "zone", "gate",
    "section", "tower", "platform", "floor", "seat", "berth", "coach",
    "season", "episode", "chapter", "class", "grade", "level", "article",
    "position", "page", "question", "step", "stage", "round", "phase",
    "wing", "block", "lane", "screen", "stand", "row", "table", "lesson",
    "unit", "part", "volume", "rank", "standard", "semester", "waitlist",
}

_STREET_WORDS = {
    "à¤®à¤¾à¤°à¥à¤—", "à¤°à¥‹à¤¡", "à¤¨à¤—à¤°", "à¤¸à¤¡à¤¼à¤•", "à¤—à¤²à¥€", "à¤šà¥Œà¤•",
    "road", "street", "lane", "avenue", "marg", "nagar", "chowk", "colony",
}

_CURRENCY_SYMBOLS = {"â‚¹": "rupee", "$": "dollar", "â‚¬": "euro", "Â£": "pound",
                     "Rs.": "rupee", "Rs": "rupee"}

_MATH_WORDS_EN = {"squared", "cubed", "plus", "minus", "times", "equals",
                  "power", "root", "over", "by", "divided", "into"}


def _fmt(template, **kw):
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


class Normalizer:
    def __init__(self, lang, *, date_order="DMY", two_digit_year_base=None):
        self.lang = lang
        self.date_order = date_order
        self.two_digit_year_base = two_digit_year_base
        self._build_regexes()

    # ------------------------------------------------------------------
    # Regex construction (depends on language vocab)
    # ------------------------------------------------------------------
    def _build_regexes(self):
        L = self.lang

        def alt(words):
            return "|".join(re.escape(w) for w in
                            sorted(set(words), key=len, reverse=True)) or r"(?!x)x"

        self._period_re = alt(list(L.PERIOD_DETECT.keys()))
        self._marker_re = alt(L.TIME_MARKERS)
        self._month_lookup = _month_lookup(L)
        self._month_re = alt(self._month_lookup)
        self._scale_re = alt(list(L.SCALE_WORDS.keys()) +
                             ["crore", "crores", "Crore", "Crores", "Cr", "cr",
                              "lakh", "lakhs", "Lakh", "Lakhs", "thousand"] +
                             sorted(_EN_SCALES))
        self._ord_suffix_re = alt(list(L.ORDINAL_INPUT_SUFFIXES) +
                                  ["st", "nd", "rd", "th"])
        native_currency = [word for word, key in L.CURRENCY_WORDS.items()
                           if not word.isascii() and key in {"rupee", "dollar", "euro", "pound"}]
        self._currency_re = r"â‚¹|\$|â‚¬|Â£|Rs\.?|INR|USD|EUR|GBP"
        if native_currency:
            self._currency_re += "|(?:" + alt(native_currency) + r")\.?"
        self._id_keywords = _ID_KEYWORDS_EN | _OTP_INPUT_WORDS | {w.lower() for w in L.ID_KEYWORDS}
        self._label_words = _LABEL_WORDS_EN | {w.lower() for w in L.LABEL_WORDS}
        self._minute_re = alt(L.MINUTE_WORDS + ["minute", "minutes", "min"])
        # ratio detection keywords: english words + the language's own ratio
        # word extracted from its RATIO template ("{a} à¤…à¤¨à¥à¤ªà¤¾à¤¤ {b}" -> à¤…à¤¨à¥à¤ªà¤¾à¤¤),
        # so "à®µà®¿à®•à®¿à®¤à®®à¯ 9:15" reads as a ratio, not a time
        tpl_words = re.sub(r"\{[ab]\}", " ", L.RATIO or "").split()
        self._ratio_kws = {"ratio", "odds", "proportion", "à¤…à¤¨à¥à¤ªà¤¾à¤¤"} | \
            {w.lower() for w in tpl_words}

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _month_index(self, token):
        return self._month_lookup.get(token.strip().lower())

    def _digits_spelled(self, digits, comma=True):
        if os.environ.get("FR3_TN_LONG_DIGITS", "0") == "1" and len(digits) >= 8:
            if __package__:
                from .identifier_rules import group_digits
            else:
                from identifier_rules import group_digits
            return group_digits(digits, self.lang)
        words = [self.lang.digit(int(d)) for d in digits]
        return ", ".join(words) if comma else " ".join(words)

    def _letters_spelled(self, letters, comma=False):
        words = [self.lang.letter(c) for c in letters]
        return ", ".join(words) if comma else " ".join(words)

    def _spell_alnum(self, token, comma):
        """Spell a code segment: letters by sound, digits one by one."""
        parts = []
        for ch in token:
            if ch.isdigit():
                parts.append(self.lang.digit(int(ch)))
            elif ch.isalpha() and ch.isascii():
                parts.append(self.lang.letter(ch))
            # other chars (hyphens) become separators, dropped
        return (", " if comma else " ").join(parts)

    def _spell_model_segment(self, seg):
        """WH-1000XM5 style segment: letter runs by sound, digit runs as
        cardinal when a clean round block, else digit-by-digit; spaces."""
        out = []
        for run in re.findall(r"[A-Za-z]+|\d+", seg):
            if run.isdigit():
                n = int(run)
                if len(run) == 1 or (n % 100 == 0 and n >= 100):
                    out.append(self.lang.cardinal(n))
                else:
                    out.append(self._digits_spelled(run, comma=False))
            else:
                out.append(self._letters_spelled(run))
        return " ".join(out)

    def _number_words(self, numstr):
        """Plain number string (commas allowed) -> cardinal / decimal words."""
        if numstr.startswith(("-", "+")):
            sign = self.lang.SYM["minus" if numstr[0] == "-" else "plus"]
            return f"{sign} {self._number_words(numstr[1:])}"
        numstr = numstr.replace(",", "")
        if "." in numstr:
            ip, fp = numstr.split(".", 1)
            return self.lang.decimal(int(ip or 0), fp)
        return self.lang.cardinal(int(numstr))

    def _keyword_before(self, text, start, keywords, window=3):
        before = _sentence_before(text, start).replace("\u200c", "")
        toks = re.findall(r"[\w#à¤€-à·¿']+", before.lower())
        for tok in toks[-window:]:
            tok = tok.strip("#'")
            if tok in keywords:
                return True
        return False

    def _period_for_hour24(self, h):
        if 4 <= h < 12:
            return "morning"
        if 12 <= h < 16:
            return "afternoon"
        if 16 <= h < 20:
            return "evening"
        return "night"

    def _time_words(self, h, m, period_key):
        if self.lang.code != "en" and (h == 0 or h > 12):
            # Keep the written 24-hour hour. The inherited language time methods
            # otherwise fold zero into twelve or fourteen into two.
            template = self.lang.TIME_HM if m else self.lang.TIME_ON_HOUR
            return template.format(h=self.lang.cardinal(h), m=self.lang.cardinal(m))
        return self.lang.time_phrase(h, m, period_key)

    # ------------------------------------------------------------------
    # Pre-clean
    # ------------------------------------------------------------------
    def _preclean(self, text):
        text = _ascii_digits(text)
        text = text.replace("âˆ’", "-")
        text = text.translate(_ZW_STRIP)
        # double quotation marks (incl. CSV-style "" and """ runs) carry no
        # sound â€” a voice model cannot speak them.
        text = re.sub(r'["â€œâ€â€ž]+', "", text)
        # single quotes wrapping phrases ('Premium Home Decor') go too.
        # Kept: apostrophes inside words (D'Souza, John's) and quoted single
        # letters ('Y' -> letter name, handled by its own matcher).
        text = re.sub(r"(?<=[\s(à¥¤,â€”â€“-])['â€˜â€™](?![A-Za-z]['â€˜â€™])(?=\S)", "", text)
        text = re.sub(r"^['â€˜â€™](?![A-Za-z]['â€˜â€™])(?=\S)", "", text)
        text = re.sub(r"(?<=\S)(?<!['â€˜â€™][A-Za-z])['â€˜â€™](?=[\s,à¥¤.!?;:)]|$)", "", text)
        text = re.sub(r"#{2,}", " ", text)   # markdown header/### junk
        # ellipses (â€¦ / ... / ..) are a spoken pause -> read as a comma
        text = re.sub(r"\s*(?:â€¦|â€¥|\.{2,})\s*", ", ", text)
        # parentheses are silent; the aside becomes a comma pause
        text = re.sub(r"\s*[()]\s*", ", ", text)
        # //comment// markers are markup junk ((?<!:) keeps https:// intact)
        text = re.sub(r"(?<!:)/{2,}", " ", text)
        # "No." / "no." before digits is the number abbreviation, not "no"
        if self.lang.NUMBER_WORD:
            text = re.sub(r"\b[Nn][Oo]s?\.\s*(?=\d)",
                          self.lang.NUMBER_WORD + " ", text)
        # written-only unit abbreviations -> native words (mins -> à°¨à°¿à°®à°¿à°·à°¾à°²à±);
        # guards keep code segments like MIN-22 intact
        for ab, word in self.lang.ABBREV_UNITS.items():
            # consume the abbreviation's own full stop when the sentence
            # continues ("Dr. Rao" -> "doctor Rao"), but keep it when the
            # abbreviation ends the sentence ("... approx." -> "approximately.")
            text = re.sub(rf"(?<![\w-]){ab}(?:\.(?=\s+\S))?(?![\w-])", word, text,
                          flags=re.IGNORECASE)
        text = re.sub(r"^[,\s]+", "", text)
        # Comparison operators are meaningful even without a left numeric operand.
        text = _EMOJI_RE.sub(" ", text)
        # en/em dash between numbers -> canonical range dash; others deleted
        text = re.sub(r"(?<=[\d])\s*[â€“â€”]\s*(?=[\d])", "â€“", text)
        # time-range dash: "11 AM â€“ 3 PM" (dash not directly digit-adjacent)
        text = re.sub(r"\s+[â€“â€”]\s+(?=\d)", " â€“ ", text)
        text = re.sub(r"[â€”]", " ", text)
        text = re.sub(r"(?<![\d\s])[â€“](?![\d\s])", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text

    # ------------------------------------------------------------------
    # Matchers â€” each returns list of (start, end, replacement)
    # Priority = order in self._matchers (earlier wins).
    # ------------------------------------------------------------------
    def _gather(self, text):
        matches = []  # (priority, start, end, replacement)
        matchers = [
            self._m_email,
            self._m_url,
            self._m_handle,
            self._m_comparison,
            self._m_time_range,
            self._m_duration,
            self._m_time_verbalized,
            self._m_time,
            self._m_tamil_date,
            self._m_date,
            self._m_currency_range,
            self._m_currency,
            self._m_bare_rupee_suffix,
            self._m_long_digit_runs,
            self._m_amount_context,
            self._m_percent_range,
            self._m_percent,
            self._m_negative,
            self._m_designation,
            self._m_vehicle_plate,
            self._m_ordinal,
            self._m_code,
            self._m_id_digits,
            self._m_chem,
            self._m_math,
            self._m_num_range,
            self._m_dimensions,
            self._m_roman,
            self._m_unit_initialism,
            self._m_kmb_suffix,
            self._m_fraction,
            self._m_ratio,
            self._m_label,
            self._m_acronym_model,
            self._m_num_hyphen_word,
            self._m_decade,
            self._m_year,
            self._m_quoted_letter,
            self._m_short_mixed_token,
            self._m_decimal,
            self._m_int,
        ]
        for prio, fn in enumerate(matchers):
            for (s, e, rep) in fn(text):
                matches.append((prio, s, e, rep))
        return matches

    # --- email / url / handle -----------------------------------------
    def _spell_uri_part(self, part):
        """Word chunks stay; digits spelled; symbols to words."""
        if os.environ.get("FR3_TN_URL_SPELL", "0") == "1":
            if __package__:
                from .identifier_rules import spell_uri
            else:
                from identifier_rules import spell_uri
            return spell_uri(part, self.lang, self._digits_spelled)
        S = self.lang.SYM
        out = []
        for run in re.findall(r"[^\W\d_]+|\d+|[^\w\s]|_", part):
            if run.isdigit():
                out.append(self._digits_spelled(run, comma=False))
            elif run == ".":
                out.append(S["dot"])
            elif run == "_":
                out.append(S["underscore"])
            elif run == "@":
                out.append(S["at"])
            elif run == "/":
                out.append(S["slash"])
            elif run == "-":
                out.append("dash")
            elif run == "+":
                out.append(S["plus"])
            elif run in {"?", "=", "&", "#", "%", ":", "~"}:
                out.append({"?": "question mark", "=": "equals", "&": "ampersand",
                            "#": "hash", "%": "percent", ":": "colon", "~": "tilde"}[run])
            elif run.lower() == "www":
                out.append(self._letters_spelled("WWW"))
            elif len(run) == 1 or run.upper() in self.lang.ABBREV_UNITS:
                # Do not expand /Dr/ to /doctor/ on a second normalization pass.
                out.append(self._letters_spelled(run.upper()))
            else:
                out.append(run)
        return " ".join(w for w in out if w)

    def _m_email(self, text):
        for m in re.finditer(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", text):
            yield m.start(), m.end(), self._spell_uri_part(m.group(0))

    def _m_url(self, text):
        pat = (r"https?://[^\s<>]+|www\.[^\s<>]+"
               # bare domains: maya.ai, irctc.co.in/booking
               r"|\b[A-Za-z][\w\-]*\.(?:com|org|net|ai|io|co|dev|app|edu|gov|in)"
               r"(?:\.(?:in|uk|us))?(?:[/?#][^\s<>]*)?(?![\w.])")
        for m in re.finditer(pat, text):
            token = m.group(0).rstrip(".,!?;)")
            body = re.sub(r"^https?://", "", token)
            yield m.start(), m.start() + len(token), self._spell_uri_part(body)

    def _m_handle(self, text):
        # trailing \w so a sentence-final period after the handle is not
        # swallowed and spoken as "dot" ("@test_user." -> handle + full stop)
        pat = r"(?<![\w.])@([A-Za-z](?:[\w.\-]*\w)?)" if os.environ.get("FR3_TN_URL_SPELL", "0") == "1" else r"(?<![\w.])@([A-Za-z](?:[\w.]*\w)?)"
        for m in re.finditer(pat, text):
            yield m.start(), m.end(), \
                f"{self.lang.SYM['at']} {self._spell_uri_part(m.group(1))}"

    # --- time -----------------------------------------------------------
    def _parse_time_match(self, m, text):
        """Shared logic. Returns (h12, mm, period_key)."""
        pass

    def _time_pattern(self):
        P = self._period_re
        MK = self._marker_re
        # (?<!\w) / (?!\w) around native-word alternations: "àª—à«àªœàª°àª¾àª¤" ends
        # with àª°àª¾àª¤ (night) â€” without boundaries the period word matches
        # inside other words
        return re.compile(
            rf"(?:(?<!\w)({P})\s+)?"                 # leading period word
            rf"(\d{{1,2}})"                          # hour
            # a real time needs :mm, AM/PM or a time marker right after the
            # hour â€” otherwise bare numbers spuriously swallow period words
            rf"(?=:\d{{2}}"
            rf"|\.\d{{2}}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)(?![A-Za-z])"   # 6.45 pm â€” dot form
            rf"|\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)(?![A-Za-z])|\s*(?:{MK})(?!\w))"
            rf"(?:[.:](\d{{2}}))?"                   # :mm  or  .mm (gated by the lookahead)
            rf"(?:\s*(AM|PM|am|pm|a\.m\.|p\.m\.))?"  # am/pm
            rf"(?:\s*({MK})(?!\w))?"                 # à¤¬à¤œà¥‡ / à¤®à¤£à¤¿ ...
            rf"(?:\s+({P})(?!\w))?", re.I           # trailing period word
        )

    def _resolve_time(self, lead_p, h, mm, ampm, marker, trail_p):
        L = self.lang
        h = int(h)
        m = int(mm) if mm is not None else 0
        period = None
        src_period = lead_p or trail_p
        if src_period:
            period = L.PERIOD_DETECT.get(src_period)
        if L.code != "en" and not ampm and not src_period:
            return h, m, None
        infer = not L.AMPM_WORDS      # languages that speak am/pm keep the token instead
        if ampm:
            a = ampm.lower().replace(".", "")
            h24 = h % 12 + (12 if a == "pm" else 0)
            if infer:
                period = period or self._period_for_hour24(h24)
        elif h > 12 or h == 0:
            # 24-hour clock
            if infer:
                period = period or self._period_for_hour24(h)
        if L.AMPM_WORDS and not ampm and (h > 12 or h == 0):
            return h, m, period      # "14:30" -> "fourteen thirty", not "two thirty"
        h12 = h % 12
        if h12 == 0:
            h12 = 12
        return h12, m, period

    def _m_duration(self, text):
        # H:MM:SS is a duration, not a 12-hour clock. Zero hours must not become 12.
        L = self.lang
        sec_word = L.SYM.get("second", "seconds")
        for m in re.finditer(r"\b(\d{1,3}):(\d{2}):(\d{2})\b", text):
            h, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if mm > 59 or ss > 59:
                continue
            hours = L.ABBREV_UNITS.get("HRS", "hours")
            minutes = L.SYM["minute"]
            parts = [L.join_amount(L.cardinal(n), unit if n != 1 or L.code != 'en' else unit.rstrip('s'))
                     for n, unit in ((h, hours), (mm, minutes), (ss, sec_word)) if n]
            yield m.start(), m.end(), " ".join(parts) or L.join_amount(L.cardinal(0), sec_word)

    def _m_time_verbalized(self, text):
        # "9 à¤¬à¤œà¤•à¤° 15 à¤®à¤¿à¤¨à¤Ÿ" â€” already half-verbalized with digits: rebuild the
        # canonical phrase instead of reading the pieces separately
        L = self.lang
        MIN = self._minute_re
        pat = re.compile(
            rf"\b(\d{{1,2}})\s*(?:{self._marker_re})\s+(\d{{1,2}})\s*({MIN})(?!\w)")
        for m in pat.finditer(text):
            h, mm = int(m.group(1)), int(m.group(2))
            if h > 23 or mm > 59:
                continue
            rep = self._time_words(h, mm, None)
            sfx = L.MARKER_SUFFIXES.get(m.group(3) or "")
            if sfx:
                rep = L.attach_postposition(rep, sfx) or rep
            yield m.start(), m.end(), rep

    def _m_time(self, text):
        pat = self._time_pattern()
        for m in pat.finditer(text):
            lead_p, h, mm, ampm, marker, trail_p = m.groups()
            if h is None:
                continue
            # must look like a time: needs marker, am/pm, period word, or a
            # valid HH:MM
            hh = int(h)
            is_hhmm = mm is not None and int(mm) <= 59 and hh < 24
            if not (ampm or marker or ((lead_p or trail_p) and (is_hhmm or mm is None))):
                if not (is_hhmm and mm is not None and hh > 12):
                    # bare h:mm with hh<=12: still a time if mm is 2-digit <=59
                    if not (is_hhmm and not self._keyword_before(
                            text, m.start(), self._ratio_kws, window=4)):
                        continue
            if mm is None and not (ampm or marker):
                continue  # bare number is not a time
            if mm is not None and int(mm) > 59:
                continue
            if hh > 23:
                continue
            h12, mnt, period = self._resolve_time(lead_p, h, mm, ampm, marker, trail_p)
            src_period = lead_p or trail_p
            if src_period:
                rep = f"{src_period} {self.lang.time_phrase(h12, mnt, None)}"
            else:
                rep = self._time_words(h12, mnt, period)
            if ampm and self.lang.AMPM_WORDS:
                w = self.lang.AMPM_WORDS.get(ampm.lower().replace(".", ""))
                if w:
                    # a bare hour with am/pm reads "eight p m", never
                    # "eight o'clock p m" -- o'clock was not in the input
                    if mnt == 0:
                        rep = self.lang.two_digit(h12)
                    rep = f"{rep} {w}"
            sfx = self.lang.MARKER_SUFFIXES.get(marker or "")
            if sfx:
                rep = self.lang.attach_postposition(rep, sfx) or rep
            yield m.start(), m.end(), rep

    def _m_time_range(self, text):
        # 11 AM â€“ 3 PM  /  9:00 â€“ 17:00  / 11 AM to 3 PM stays "to"
        pat = self._time_pattern()
        rng = re.compile(
            rf"({pat.pattern})\s*[â€“\-]\s*({pat.pattern})", pat.flags)
        for m in rng.finditer(text):
            g = m.groups()
            left, right = g[0], g[7]
            gl, gr = g[1:7], g[8:14]
            if gl[1] is None or gr[1] is None:
                continue
            # each side must independently look like a time
            if not ((gl[3] or gl[4] or gl[2]) and (gr[3] or gr[4] or gr[2])):
                continue
            h1, m1, p1 = self._resolve_time(*gl)
            h2, m2, p2 = self._resolve_time(*gr)
            a = self._time_words(h1, m1, p1)
            b = self._time_words(h2, m2, p2)
            if self.lang.AMPM_WORDS:
                if gl[3]:
                    a = (self.lang.two_digit(h1) if m1 == 0 else a) + " " + self.lang.AMPM_WORDS[gl[3].lower().replace('.', '')]
                if gr[3]:
                    b = (self.lang.two_digit(h2) if m2 == 0 else b) + " " + self.lang.AMPM_WORDS[gr[3].lower().replace('.', '')]
            rep = _fmt(self.lang.RANGE_TO, a=a, b=b)
            tail = self.lang.RANGE_TO.split("{b}")[-1].strip()
            if tail and re.match(rf"\s*{re.escape(tail)}(?!\w)",
                                 text[m.end():]):
                rep = rep[: -len(tail)].rstrip()  # source already says it
            yield m.start(), m.end(), rep

    # --- dates -----------------------------------------------------------
    def _m_tamil_date(self, text):
        if self.lang.code not in ("en", "ta"):
            return
        months = "|".join(_TAMIL_MONTHS)
        # Only calendar-day suffixes. Do not pretend all Tamil number morphology
        # can be repaired by concatenating an English cardinal with a suffix.
        pat = rf"(?<!\w)({months})\s+(\d{{1,2}})(?:-?(?:à®†à®®à¯|à®®à¯))?(?![\dA-Za-z])(?:\s*à®¤à¯‡à®¤à®¿)?"
        for m in re.finditer(pat, text):
            d = int(m[2])
            if 1 <= d <= 31:
                yield m.start(), m.end(), f"{m[1]} {_TAMIL_ORDINALS[d]} à®¤à¯‡à®¤à®¿"
        for m in re.finditer(r"(?<!\w)(\d+)-?(?:à®†à®®à¯|à®®à¯)(?![\w\u0B80-\u0BFF])", text):
            d = int(m[1])
            if not 1 <= d <= 31:
                if self.lang.code == "en":
                    raise NormalizationError("unsupported_tamil_ordinal", "Select Tamil normalization for ordinals outside calendar days 1-31.", span=m.span(), source=text)
                yield m.start(), m.end(), self.lang.ordinal(d, "à®†à®®à¯")
            else:
                yield m.start(), m.end(), _TAMIL_ORDINALS[d]

    def _date_words(self, day, month, year=None, ordinal=False, month_first=False):
        L = self.lang
        if L.code == "ta":
            out = f"{L.month(month)} {_TAMIL_ORDINALS[day]} à®¤à¯‡à®¤à®¿"
        else:
            suffix = ordinal if isinstance(ordinal, str) and ordinal in L.ORDINAL_INPUT_SUFFIXES else ""
            day_words = L.ordinal(day, suffix) if ordinal else L.cardinal(day)
            out = f"{L.month(month)} {day_words}" if month_first and L.code != "en" else f"{day_words} {L.month(month)}"
        return f"{out} {L.year(year)}" if year is not None else out

    def _m_date(self, text):
        L = self.lang
        for m in re.finditer(rf"\b(\d{{1,2}})[-â€“](\d{{1,2}})\s+({self._month_re})(?!\w)(?:\s+(\d{{4}}))?", text, re.I):
            mo = self._month_index(m[3])
            for day in (m[1], m[2]):
                _calendar_check(int(day), mo, int(m[4]) if m[4] else 2000)
            days = _fmt(L.RANGE, a=L.cardinal(int(m[1])), b=L.cardinal(int(m[2])))
            out = f"{days} {L.month(mo)}"
            if m[4]:
                out += f" {L.year(int(m[4]))}"
            yield m.start(), m.end(), out
        for m in re.finditer(r"(?<![\w-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])", text):
            yield m.start(), m.end(), self._date_words(int(m[3]), int(m[2]), int(m[1]))
        # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
        for m in re.finditer(r"\b(\d{1,2})([/.\-])(\d{1,2})\2(\d{2,4})\b(?![/A-Za-z-])", text):
            d, mo, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
            if self.date_order == "MDY":
                d, mo = mo, d
            if not (1 <= d <= 31 and 1 <= mo <= 12):
                continue
            if len(m.group(4)) == 2:
                y += self.two_digit_year_base
            yield m.start(), m.end(), self._date_words(d, mo, y)
        # DD-Mon-YYYY / DD Mon YYYY / DD Mon, with native or English months
        # (?!\w) after the month: without it "25 à²®à³‡à²²à³†" would parse as
        # "25 May" because à²®à³‡à²²à³† starts with à²®à³‡ (May)
        pat = re.compile(
            rf"\b(\d{{1,2}})(?:-?({self._ord_suffix_re}))?[\-\s]({self._month_re})(?!\w)"
            rf"(?:[\-\s,]+(\d{{4}}))?", re.IGNORECASE)
        for m in pat.finditer(text):
            d = int(m.group(1))
            mo = self._month_index(m.group(3))
            if mo is None or not (1 <= d <= 31):
                continue
            out = self._date_words(d, mo, int(m[4]) if m[4] else None, m[2])
            yield m.start(), m.end(), out
        # Month DD, YYYY (English order)
        pat2 = re.compile(
            rf"\b({self._month_re})\s+(\d{{1,2}})(?:-?({self._ord_suffix_re}))?(?:,?\s+(\d{{4}}))?(?!\d)",
            re.IGNORECASE)
        for m in pat2.finditer(text):
            mo = self._month_index(m.group(1))
            d = int(m.group(2))
            if mo is None or not (1 <= d <= 31):
                continue
            out = self._date_words(d, mo, int(m[4]) if m[4] else None, m[3], month_first=True)
            yield m.start(), m.end(), out

    # --- currency ---------------------------------------------------------
    _CURRENCY_CODE_KEYS = {"â‚¹": "rupee", "$": "dollar", "â‚¬": "euro",
                           "Â£": "pound", "RS": "rupee", "INR": "rupee",
                           "USD": "dollar", "EUR": "euro", "GBP": "pound"}

    def _cur_key(self, symtok):
        t = symtok.strip().rstrip(".").strip().upper()
        return self._CURRENCY_CODE_KEYS.get(t) or self.lang.CURRENCY_WORDS.get(t) or self.lang.CURRENCY_WORDS.get(t + ".") or "rupee"

    def _m_currency_range(self, text):
        # â‚¹500-700 -> "five hundred to seven hundred rupees"
        L = self.lang
        pat = re.compile(
            rf"(?<!\w)([+-]?)({self._currency_re})"
            rf"\s*([+-]?\d+(?:,\d{{2,3}})*(?:\.\d+)?)\s*[â€“\-]\s*([+-]?\d+(?:,\d{{2,3}})*(?:\.\d+)?)(?!\d|[.,]\d)", re.I)
        for m in pat.finditer(text):
            if m[1] and m[3].startswith(("-", "+")):
                raise NormalizationError("conflicting_sign", "Currency amount has two signs; confirm the value.", span=m.span(), source=text)
            a, b = self._number_words(m[1] + m[3]), self._number_words(m[4])
            cw = L.SYM[self._cur_key(m[2])]
            rng = _fmt(L.RANGE, a=a, b=L.join_amount(b, cw))
            yield m.start(), m.end(), rng

    def _m_bare_rupee_suffix(self, text):
        L = self.lang
        for m in re.finditer(r"(?<![\wâ‚¹.])(\d+(?:,\d{2,3})*(?:\.\d+)?)\s?/-", text):
            num = m.group(1).replace(",", "")
            words = self._number_words(num)
            yield m.start(), m.end(), L.join_amount(words, L.SYM["rupee"])

    def _m_currency(self, text):
        L = self.lang
        S = L.SYM
        pat = re.compile(
            rf"(?<!\d)([+-]?)\s*(â‚¹|\$|â‚¬|Â£|(?<!\w)(?:{self._currency_re}))"
            rf"\s*([+-]?)(\d+(?:,\d{{2,3}})*(?:\.\d+)?)"
            rf"(?:\s*({self._scale_re})(?:((?:à¤¾à¤‚|à¤‚)?(?:à¤šà¤¾|à¤šà¥€|à¤šà¥‡|à¤šà¥à¤¯à¤¾|à¤¤|à¤¨à¤¾|à¤¹à¥‚à¤¨|à¤ªà¤°à¥à¤¯à¤‚à¤¤|à¤®à¤§à¥à¤¯à¥‡))|(?![à¤€-à¤ƒà¤º-à¥à¦€-à¦ƒà¦¾-à§à¨-à¨ƒà¨¾-à©àª-àªƒàª¾-à«à¬-à¬ƒà¬¾-à­à®‚à®¾-à¯à°€-à°„à°¾-à±Œà²-à²ƒà²¾-à³Œà´€-à´ƒà´¾-àµŒ]))(?![A-Za-z0-9]))?(?:\s?/-)?", re.I)
        for m in pat.finditer(text):
            if m[1] and m[3]:
                raise NormalizationError("conflicting_sign", "Currency amount has two signs; confirm the value.", span=m.span(), source=text)
            key = self._cur_key(m[2])
            num = m[4].replace(",", "")
            scale = m[5]
            cur_word = S[key]
            # money with exactly 2 decimals and no scale word -> main + sub
            # unit ($84.20 -> 84 dollar 20 cent; â‚¹95.41 -> 95 rupees 41 paise)
            if "." in num and not scale and len(num.split(".", 1)[1]) == 2:
                ip, fp = num.split(".", 1)
                main = L.cardinal(int(ip or 0))
                if key == "rupee" and L.SYM.get("paisa"):
                    sub_word = L.SYM["paisa"]
                else:
                    sub_word = S["cent"] if key != "rupee" else None
                if fp.strip("0") == "":
                    out = L.join_amount(main, cur_word)
                elif sub_word:
                    out = (f"{L.join_amount(main, cur_word)} "
                           f"{L.join_amount(L.cardinal(int(fp)), sub_word)}")
                else:
                    out = L.join_amount(L.decimal(int(ip or 0), fp), cur_word)
            else:
                words = self._number_words(num)
                if scale:
                    sl = scale.lower()
                    canon = L.SCALE_WORDS.get(scale)
                    if canon is None and sl in _EN_SCALES:
                        scale_word = scale  # 1.5 million -> ... million dollar
                    else:
                        if canon is None:
                            canon = ("crore" if sl in ("cr",) or "crore" in sl
                                     else "lakh" if "lakh" in sl
                                     else "thousand")
                        scale_word = L.scale_noun(canon, num)
                    out = L.join_amount(L.join_amount(words, scale_word),
                                        cur_word)
                    case = m[6]
                    if case:
                        out = L.attach_postposition(out, case) or out
                else:
                    out = L.join_amount(words, cur_word)
            sign = m[1] or m[3]
            if sign:
                out = f"{S['minus'] if sign == '-' else S['plus']} {out}"
            # Keep whitespace outside the matched amount.
            space = m[0][:1].isspace() or (m.start() > 0 and text[m.start() - 1].isalnum())
            yield m.start(), m.end(), (" " if space else "") + out

    def _m_amount_context(self, text):
        # Explicit currency suffix/amount label wins over broad identifier hints.
        units = (r"rupees?|rs\.?|inr|dollars?|usd|euros?|pounds?|paise|paisa|cents?|"
                 r"à®°à¯‚à®ªà®¾à®¯à¯|à®ªà¯ˆà®šà®¾|à¤°à¥à¤ªà¤¯à¥‡|à¤°à¥à¤ªà¤¯à¤¾|à¤°à¥à¤ªà¤|à°°à±‚\.?|à°°à±‚à°ªà°¾à°¯à°²à±|à´°àµ‚à´ª")
        units += "|" + "|".join(map(re.escape, self.lang.CURRENCY_WORDS))
        for m in re.finditer(rf"(?<![\w.])([+-]?\d[\d,]*(?:\.\d+)?)(?=\s*(?:{units})(?![A-Za-z]))", text, re.I):
            yield m.start(), m.end(), self._number_words(m[1])
        labels = {"balance", "amount", "payable", "outstanding", "payment", "price", "fee", "pay", "emi", "à®¤à®µà®£à¯ˆ"}
        for m in re.finditer(r"(?<![\w.,+/-])\d+(?:,\d{2,3})*(?!\w|[.,/-]\d)", text):
            before = _sentence_before(text, m.start()).lower()
            toks = re.findall(r"\w+", before)[-3:]
            if any(t in labels for t in toks) and not any(t in {"number", "id", "code", "otp", "pin"} for t in toks):
                yield m.start(), m.end(), self._number_words(m[0])

    # --- percent ------------------------------------------------------------
    def _m_percent_range(self, text):
        # 40-70% / 40â€“70% -> "40 to 70 percent"
        L = self.lang
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*[â€“\-]\s*(\d+(?:\.\d+)?)\s*%",
                             text):
            a = self._number_words(m.group(1))
            b = self._number_words(m.group(2))
            yield m.start(), m.end(), \
                f"{_fmt(L.RANGE, a=a, b=b)} {L.SYM['percent']}"

    def _m_percent(self, text):
        L = self.lang
        for m in re.finditer(r"(-?)(\d+(?:,\d{2,3})*(?:\.\d+)?)\s*%", text):
            words = self._number_words(m.group(2))
            if m.group(1):
                words = f"{L.SYM['minus']} {words}"
            yield m.start(), m.end(), f"{words} {L.SYM['percent']}"

    # --- designations (Sony WH-1000XM5, Kepler-186f, Soyuz T-11) -----------
    def _m_designation(self, text):
        # TitleCase brand fused: Kepler-186f, Salyut-7
        for m in re.finditer(r"\b([A-Z][a-z]{2,})-([A-Za-z0-9]+)\b", text):
            seg = m.group(2)
            if not re.search(r"\d", seg):
                continue
            if self._month_index(seg) or self._month_index(m.group(1)):
                continue
            yield m.start(), m.end(), \
                f"{m.group(1)} {self._spell_model_segment(seg)}"
        # Brand word + short designation: Sony WH-1000XM5, Soyuz T-11
        pat = re.compile(
            r"\b([A-Z][a-z]{2,})\s+([A-Z]{1,4}-[A-Za-z0-9]+)\b")
        for m in pat.finditer(text):
            brand = m.group(1)
            if brand.lower() in self._id_keywords or brand.lower() in self._label_words:
                continue
            seg = m.group(2)
            if not re.search(r"\d", seg):
                continue
            yield (m.start(2), m.end(2), self._spell_model_segment(seg))

    # --- alphanumeric codes -------------------------------------------------
    def _m_code(self, text):
        # hyphenated tokens containing at least one letter and one digit
        pat = re.compile(r"#?\b(?=[A-Za-z0-9/-]*\d)(?=[A-Za-z0-9/-]*[A-Za-z])"
                         r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+(?![A-Za-z0-9])")
        for m in pat.finditer(text):
            tok = m.group(0).lstrip("#")
            # date like 12-Aug-2025 -> handled by date matcher (higher prio)
            parts = re.split(r"[-/]", tok)
            if (any(self._month_index(p) for p in parts)
                    and all(p.isdigit() for p in parts
                            if not self._month_index(p))):
                continue  # date-like 12-Aug-2025; MAY-17Y7 is still a code
            # a lowercase word segment means this is prose, not a code
            # (900-point, 80-watt) -> handled by num-hyphen-word matcher
            if any(re.fullmatch(r"[a-z][a-z]+|[A-Z][a-z]{2,}", p) for p in parts):
                continue  # prose word segment (900-point, 7-East)
            if (len(parts) == 2 and parts[0].isdigit() and parts[1].isalpha()
                    and len(parts[1]) >= 3):
                continue  # 30-DAY, 24-HOUR -> number + word
            # 15-GB -> unit matcher
            if (len(parts) == 2 and parts[0].isdigit()
                    and parts[1].upper() in _UNIT_INITIALISMS):
                continue
            # Hall 3-A / Window 5-B -> label matcher (but CR-2A stays a code)
            if (re.fullmatch(r"\d{1,3}-?[A-Za-z]", tok)
                    and self._keyword_before(text, m.start(),
                                             self._label_words, window=2)):
                continue
            if sum(c.isalnum() for c in tok) <= 2:
                continue
            yield m.start(), m.end(), self._spell_alnum(tok, comma=True)
        # solid mixed tokens (length>=5, >=2 digits): MH12AB8821, 77K2L9
        pat2 = re.compile(r"\b(?=(?:[A-Za-z]*\d){2})(?=[A-Za-z0-9]*[A-Za-z])"
                          r"[A-Za-z0-9]{5,}\b")
        for m in pat2.finditer(text):
            tok = m.group(0)
            if re.fullmatch(r"\d+s", tok):
                continue  # 1980s -> decade matcher
            if re.fullmatch(r"\d+[xÃ—]\d+", tok):
                continue  # 1920x1080 -> dimensions matcher
            m2 = re.fullmatch(r"\d+([A-Za-z]{1,4})", tok)
            if m2 and m2.group(1).upper() in _UNIT_INITIALISMS:
                continue  # 250ml -> unit matcher
            if re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", tok) and self._chem_ok(tok):
                continue  # chemistry wins
            yield m.start(), m.end(), self._spell_alnum(tok, comma=True)

    def _m_vehicle_plate(self, text):
        # spaced Indian plates: AP 09 CL 4421 / TS 09 EA 1234
        L = self.lang
        pat = re.compile(r"\b([A-Z]{2})[ -](\d{1,2})[ -]([A-Z]{1,3})[ -](\d{3,4})\b")
        for m in pat.finditer(text):
            out = []
            for grp in m.groups():
                for ch in grp:
                    out.append(L.digit(int(ch)) if ch.isdigit() else L.letter(ch))
            yield m.start(), m.end(), ", ".join(out)

    # --- identifier digit runs ----------------------------------------------
    def _m_long_digit_runs(self, text):
        if os.environ.get("FR3_TN_LONG_DIGITS", "0") != "1":
            return
        if __package__:
            from .identifier_rules import long_runs
        else:
            from identifier_rules import long_runs
        reserved = [(s, e) for matcher in (self._m_percent_range, self._m_percent, self._m_negative)
                    for s, e, _ in matcher(text)]
        for s, e, spoken in long_runs(text, self.lang):
            if not any(s < end and e > start for start, end in reserved):
                yield s, e, spoken

    def _m_id_digits(self, text):
        L = self.lang
        # hyphen/space separated digit groups, or long runs, or # prefixed
        pat = re.compile(r"(\+?)(?:#?\b\d+(?:[-\s]\d+)+\b|#\s?\d{3,}\b"
                         r"|(?<![\d.])(?<!\d,)\d{7,}(?![.,]?\d))")
        for m in pat.finditer(text):
            has_plus = m.group(1) == "+"
            tok = m.group(0).lstrip("+# ")
            groups = re.split(r"[-\s]", tok)
            digits = "".join(groups)
            is_id = False
            if has_plus and len(digits) >= 8:
                is_id = True  # +91 ... country-code phone
            elif m.group(0).startswith("#"):
                is_id = True
            elif len(digits) >= 7 and " " not in tok:
                is_id = True
            elif any(len(g) >= 4 for g in groups) and len(groups) >= 2 and "-" in tok:
                is_id = True
            elif len(groups) >= 3 and "-" in tok:
                is_id = True
            elif self._keyword_before(text, m.start(), self._id_keywords):
                is_id = True
            else:
                a_toks = re.findall(r"[\w#à¤€-à·¿']+",
                                    _sentence_after(text, m.end()).lower())[:4]
                if any(t in self._id_keywords for t in a_toks):
                    is_id = True
            if not is_id:
                continue
            # ranges like 10-20 without keyword: ascending small pairs -> skip
            a_toks2 = re.findall(r"[\w#à¤€-à·¿']+",
                                 _sentence_after(text, m.end()).lower())[:4]
            if ("-" in tok and len(groups) == 2
                    and all(len(g) <= 3 for g in groups)
                    and int(groups[0]) < int(groups[1])
                    and not self._keyword_before(text, m.start(), self._id_keywords)
                    and not any(t in self._id_keywords for t in a_toks2)):
                continue
            rep = self._digits_spelled(digits, comma=True)
            if has_plus:
                rep = f"{L.SYM['plus']} {rep}"
            yield m.start(), m.end(), rep
        # keyword-driven short runs: OTP 4471, code 22377, train 207,
        # "ending in 7104"
        for m in re.finditer(r"(?<![\d.])(?<!\d,)\d{3,6}(?![.,]?\d)(?!s\b)", text):
            # "139 dial à¤•à¤°à¥‡à¤‚" â€” dial/à¤¡à¤¾à¤¯à¤² shortly after marks a hotline number
            after_toks = re.findall(r"[\w#à¤€-à·¿']+",
                                    _sentence_after(text, m.end()).lower())[:6]
            id_after = {"dial", "à¤¡à¤¾à¤¯à¤²"} | set(self.lang.ID_AFTER_WORDS)
            if any(t in id_after for t in after_toks):
                yield m.start(), m.end(), \
                    self._digits_spelled(m.group(0), comma=True)
                continue
            # an identifier keyword within ~4 tokens AFTER also marks it
            # ("4321 à¤¹à¤¾ OTP", "998877 à¤µà¤¾à¤ªà¤°à¤¾ password") â€” but not label/year
            if (any(t in self._id_keywords for t in after_toks[:4])
                    and not self._keyword_before(text, m.start(), self._label_words, window=2)
                    and not self._keyword_before(text, m.start(),
                             {w.lower() for w in L.YEAR_WORDS} | {"year"}, window=2)):
                yield m.start(), m.end(), \
                    self._digits_spelled(m.group(0), comma=True)
                continue
            if self._keyword_before(text, m.start(), self._id_keywords):
                # label words override (Room 314 stays cardinal)
                if self._keyword_before(text, m.start(), self._label_words, window=2):
                    continue
                # year words override (à¤¸à¤¾à¤² 2026)
                if self._keyword_before(text, m.start(),
                                        {w.lower() for w in L.YEAR_WORDS} | {"year"},
                                        window=2):
                    continue
                yield m.start(), m.end(), \
                    self._digits_spelled(m.group(0), comma=True)

    # --- chemistry ------------------------------------------------------------
    _ELEMENTS = {
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
        "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Fe", "Cu", "Zn", "Ag",
        "Au", "Hg", "Pb", "I", "Mn", "Cr", "Ni", "Sn", "Br", "Mg",
    }
    _CHEM_WHITELIST = {"HCl", "NaCl", "KCl", "NaOH", "KOH", "HBr", "HF",
                       "CO", "NO", "NH3", "CH4", "H2O", "CO2", "SO2", "O2",
                       "N2", "H2", "H2SO4", "HNO3", "CaCO3", "C6H12O6",
                       "NaHCO3", "MgSO4", "ZnO", "FeO"}

    def _chem_ok(self, tok):
        if tok in self._CHEM_WHITELIST:
            return True
        pieces = re.findall(r"[A-Z][a-z]?|\d+", tok)
        letters = [p for p in pieces if not p.isdigit()]
        digits = [p for p in pieces if p.isdigit()]
        if not letters or any(p not in self._ELEMENTS for p in letters):
            return False
        return bool(digits) and len(letters) >= 2 or (bool(digits) and tok in self._CHEM_WHITELIST)

    def _m_chem(self, text):
        L = self.lang
        for m in re.finditer(r"\b(?:[A-Z][a-z]?\d*){1,8}\b", text):
            tok = m.group(0)
            if len(tok) < 2 or tok.upper() in _ACRONYM_WHITELIST:
                continue
            if not self._chem_ok(tok):
                continue
            out = []
            for piece in re.findall(r"[A-Z][a-z]?|\d+", tok):
                if piece.isdigit():
                    out.append(L.cardinal(int(piece)))
                else:
                    out.append(self._letters_spelled(piece.upper()))
            yield m.start(), m.end(), " ".join(out)

    # --- math -------------------------------------------------------------
    def _m_comparison(self, text):
        # Indic comparative phrases are postpositions: '<500' -> '500 à¤¸à¥‡ à¤•à¤®',
        # not 'à¤¸à¥‡ à¤•à¤® 500'. Treat the numeric bound as one semantic span.
        L = self.lang
        number = r"[+-]?\d+(?:,\d{2,3})*(?:\.\d+)?"
        operand = rf"(?:â‚¹\s*)?{number}(?:\s*%)?"
        pattern = rf"(?<![\w.])(?:(?P<left>{operand})\s*)?(?P<op><=|>=|â‰¤|â‰¥|<|>)\s*(?P<right>{operand})"
        def words(token):
            s = token.strip()
            if s.startswith("â‚¹"):
                return L.join_amount(self._number_words(s[1:].strip()), L.SYM["rupee"])
            if s.endswith("%"):
                return f"{self._number_words(s[:-1].strip())} {L.SYM['percent']}"
            return self._number_words(s)
        ors = {"hi": "à¤¯à¤¾", "te": "à°²à±‡à°¦à°¾", "ta": "à®…à®²à¯à®²à®¤à¯", "bn": "à¦¬à¦¾", "gu": "àª…àª¥àªµàª¾",
               "kn": "à²…à²¥à²µà²¾", "ml": "à´…à´²àµà´²àµ†à´™àµà´•à´¿àµ½", "mr": "à¤•à¤¿à¤‚à¤µà¤¾", "or": "à¬•à¬¿à¬®à­à¬¬à¬¾", "pa": "à¨œà¨¾à¨‚"}
        for m in re.finditer(pattern, text):
            op = m['op']
            comparison = L.SYM["lt" if op.startswith("<") or op == "â‰¤" else "gt"]
            if len(op) > 1 or op in ("â‰¤", "â‰¥"):
                comparison += " or equal to" if L.code == "en" else f" {ors[L.code]} {L.SYM['equals']}"
            left = words(m['left']) + " " if m['left'] else ""
            right = words(m['right'])
            yield m.start(), m.end(), left + (f"{comparison} {right}" if L.code == "en" else f"{right} {comparison}")

    def _m_math(self, text):
        L = self.lang
        S = L.SYM
        # power: 10^23  /  x^2
        for m in re.finditer(r"(\d+(?:\.\d+)?|[A-Za-z])\s*\^\s*(\d+|[A-Za-z])", text):
            base, exp = m.group(1), m.group(2)
            bw = self._number_words(base) if base[0].isdigit() else L.letter(base)
            ew = self._number_words(exp) if exp[0].isdigit() else L.letter(exp)
            yield m.start(), m.end(), _fmt(L.POWER, base=bw, exp=ew)
        # squared Â²
        for m in re.finditer(r"(\d+(?:\.\d+)?|[A-Za-z])Â²", text):
            v = m.group(1)
            vw = self._number_words(v) if v[0].isdigit() else L.letter(v)
            yield m.start(), m.end(), f"{vw} {S['squared']}"
        # âˆš
        for m in re.finditer(r"âˆš\s*(\d+(?:\.\d+)?)", text):
            yield m.start(), m.end(), f"{S['sqrt']} {self._number_words(m.group(1))}"
        # operator chains: x + 5 = 12, E = mc^2, 17 Ã— 3 = 51 â€” rendered only
        # when a strong math operator (= Ã— Ã· ^ âˆš) is present, so hyphen
        # ranges/ids and prose like "e-mail" never match
        eq = re.compile(
            r"(?<![\w.])((?:[A-Za-z]{1,2}|\d+(?:\.\d+)?)"
            r"(?:\s*[=+\-âˆ’Ã—*Ã·/^]\s*(?:[A-Za-z]{1,2}|\d+(?:\.\d+)?))+)")
        for m in eq.finditer(text):
            span = m.group(1)
            if not re.search(r"[=Ã—Ã·^âˆš]", span):
                continue
            yield m.start(1), m.end(1), self._render_math_expr(span)
        # single letters adjacent to verbal math words: "a squared plus b"
        for m in re.finditer(r"\b([A-Za-z])\b(?=\s+(?:squared|cubed|plus|minus|"
                             r"times|equals|to the power)\b)", text):
            yield m.start(), m.end(), L.letter(m.group(1))
        # Operators are expanded in-place in _final_sweep. This also handles
        # thresholds without a left operand and preserves x < y word order.

    def _render_math_expr(self, span):
        L = self.lang
        S = L.SYM
        opmap = {"=": S["equals"], "+": S["plus"], "-": S["minus"],
                 "âˆ’": S["minus"], "Ã—": S["times"], "*": S["times"],
                 "Ã·": S["divide"], "/": S["divide"]}
        toks = re.findall(r"\d+(?:\.\d+)?|[A-Za-z]+|[=+\-âˆ’Ã—*Ã·/^]", span)
        out = []
        i = 0
        while i < len(toks):
            tok = toks[i]
            if tok == "^":
                # base already emitted; render exponent via the power template
                exp = toks[i + 1] if i + 1 < len(toks) else ""
                ew = (self._number_words(exp) if exp[:1].isdigit()
                      else self._letters_spelled(exp.upper()))
                base = out.pop() if out else ""
                out.append(_fmt(L.POWER, base=base, exp=ew))
                i += 2
                continue
            if tok in opmap:
                out.append(opmap[tok])
            elif tok[0].isdigit():
                out.append(self._number_words(tok))
            else:
                out.append(self._letters_spelled(tok.upper()))
            i += 1
        return " ".join(out)

    # --- ranges -----------------------------------------------------------
    def _m_num_range(self, text):
        L = self.lang
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*[â€“]\s*(\d+(?:\.\d+)?)", text):
            a = self._number_words(m.group(1))
            b = self._number_words(m.group(2))
            yield m.start(), m.end(), _fmt(L.RANGE, a=a, b=b)
        # hyphen range: only ascending small pairs, no id keyword
        for m in re.finditer(r"\b(\d{1,3})-(\d{1,3})\b", text):
            if self._keyword_before(text, m.start(), self._id_keywords):
                continue
            a, b = int(m.group(1)), int(m.group(2))
            if a == b:
                # 50-50 -> "fifty fifty"
                yield m.start(), m.end(), f"{L.cardinal(a)} {L.cardinal(b)}"
                continue
            if a > b:
                if re.match(r"\s+(?:days?|months?|years?|hours?|minutes?)\b", text[m.end():], re.I):
                    yield m.start(), m.end(), _fmt(L.RANGE, a=L.cardinal(a), b=L.cardinal(b))
                    continue
                # score-style pair: "3-2 à¤¸à¥‡ à¤œà¥€à¤¤à¥‡" -> "à¤¤à¥€à¤¨ à¤¦à¥‹"
                yield m.start(), m.end(), f"{L.cardinal(a)} {L.cardinal(b)}"
                continue
            yield m.start(), m.end(), \
                _fmt(L.RANGE, a=L.cardinal(a), b=L.cardinal(b))

    # --- dimensions / "x" as times: 24x7, 1920x1080, 6 x 4 -------------------
    def _m_dimensions(self, text):
        L = self.lang
        for m in re.finditer(
                r"\b(\d+(?:\.\d+)?)\s*[xÃ—]\s*(\d+(?:\.\d+)?)\b", text):
            if m.group(1) == "0":
                continue  # hex literals like 0x1F
            yield m.start(), m.end(), \
                f"{self._number_words(m.group(1))} {L.SYM['times']} " \
                f"{self._number_words(m.group(2))}"
        # lone x between something and a number (higher-priority matcher may
        # have consumed the left operand: "â‚¹4,583.33 x 24")
        for m in re.finditer(r"(?<=\s)[xÃ—](?=\s\d)", text):
            yield m.start(), m.end(), L.SYM["times"]

    # --- Roman numerals after label words: Class XII, World War II ----------
    _ROMAN_VALS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    def _roman_value(self, s):
        total = 0
        for i, ch in enumerate(s):
            v = self._ROMAN_VALS[ch]
            if i + 1 < len(s) and self._ROMAN_VALS[s[i + 1]] > v:
                total -= v
            else:
                total += v
        return total

    def _m_roman(self, text):
        L = self.lang
        # {2,7}: single letters stay letters (Counter 2-C reads "C", not 100)
        pat = re.compile(
            r"\b(?=[IVXLCDM]{2,7}\b)M{0,3}(?:CM|CD|D?C{0,3})"
            r"(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\b")
        ctx = self._label_words | {"war", "vol", "part", "act", "book",
                                   "youdh", "à¤¯à¥à¤¦à¥à¤§", "à¤…à¤§à¥à¤¯à¤¾à¤¯", "à¤•à¤•à¥à¤·à¤¾",
                                   "à¤­à¤¾à¤—", "à¤–à¤‚à¤¡"}
        for m in pat.finditer(text):
            tok = m.group(0)
            if not tok:
                continue
            # only with a label-ish word right before â€” avoids XL sizes,
            # initials, and words like "MIX"
            if not self._keyword_before(text, m.start(), ctx, window=2):
                continue
            yield m.start(), m.end(), L.cardinal(self._roman_value(tok))

    # --- K/M/B suffixes glued to numbers: 3.4K, 1.2M -------------------------
    def _m_kmb_suffix(self, text):
        L = self.lang
        for m in re.finditer(r"(?<![\d.A-Za-z])(\d+(?:\.\d+)?)([KMB])\b", text):
            yield m.start(), m.end(), \
                f"{self._number_words(m.group(1))} {L.letter(m.group(2))}"

    # --- ordinals -----------------------------------------------------------
    def _m_ordinal(self, text):
        L = self.lang
        pat = re.compile(rf"\b(\d+)-?({self._ord_suffix_re})(?![A-Za-z0-9])", re.I)
        for m in pat.finditer(text):
            n = int(m.group(1))
            suf = m.group(2).lower()
            if suf in ("st", "nd", "rd", "th"):
                yield m.start(), m.end(), L.ordinal(n)
            else:
                yield m.start(), m.end(), L.ordinal(n, suf)

    # --- fractions -----------------------------------------------------------
    def _m_fraction(self, text):
        L = self.lang
        S = L.SYM
        pat = re.compile(r"\b(\d{1,4}(?:\.\d+)?)/(\d{1,4}(?:\.\d+)?)\b")
        for m in pat.finditer(text):
            a, b = m.group(1), m.group(2)
            if (a, b) == ("1", "2"):
                rep = S["half"]
            elif (a, b) == ("1", "4"):
                rep = S["quarter"]
            elif (a, b) == ("3", "4"):
                rep = S["three_quarters"]
            else:
                rep = _fmt(L.FRACTION, num=self._number_words(a),
                           den=self._number_words(b))
            yield m.start(), m.end(), rep

    # --- ratios -----------------------------------------------------------
    def _m_ratio(self, text):
        L = self.lang
        for m in re.finditer(r"\b(\d{1,3}):(\d{1,2})\b", text):
            mm = m.group(2)
            # 2-digit right side <=59 looks like a time -> time matcher takes it
            if len(mm) == 2 and int(mm) <= 59 and not self._keyword_before(
                    text, m.start(), self._ratio_kws, window=4):
                continue
            after = text[m.end():m.end() + 20].strip()
            if any(after.startswith(w) for w in self._ratio_kws if not
                   w.isascii()):
                # source already says the ratio word ("1:3 à°¨à°¿à°·à±à°ªà°¤à±à°¤à°¿à°²à±‹")
                yield m.start(), m.end(), \
                    f"{L.cardinal(int(m.group(1)))} {L.cardinal(int(mm))}"
                continue
            yield m.start(), m.end(), \
                _fmt(L.RATIO, a=L.cardinal(int(m.group(1))),
                     b=L.cardinal(int(mm)))

    # --- labels (Room 314, Hall 3-A, Seat 23A, coach B4) --------------------
    def _m_label(self, text):
        L = self.lang
        lbl = "|".join(re.escape(w) for w in sorted(self._label_words, key=len,
                                                    reverse=True))
        pat = re.compile(
            rf"\b({lbl})\s+(?:(\d+)(?![\d.]|,\d)(?!-[A-Za-z]{{2,}})(?:[-\s]?([A-Za-z])\b)?|([A-Za-z])\b(?:[-\s]?(\d+))?)",
            re.IGNORECASE)
        for m in pat.finditer(text):
            word = m.group(1)
            out = [word]
            if m.group(2):
                out.append(L.cardinal(int(m.group(2))))
                if m.group(3):
                    out.append(L.letter(m.group(3)))
            elif m.group(4):
                if len(m.group(4)) == 1:
                    out.append(L.letter(m.group(4)))
                else:
                    out.append(m.group(4))
                if m.group(5):
                    out.append(L.cardinal(int(m.group(5))))
            yield m.start(), m.end(), " ".join(out)

    # --- ALLCAPS model prefix + number: "XPS 15" ----------------------------
    def _m_acronym_model(self, text):
        L = self.lang
        for m in re.finditer(r"\b([A-Z]{2,4})\s+(\d{1,3})(?![\d.])", text):
            tok = m.group(1)
            if tok in _ACRONYM_WHITELIST or tok.upper() in _UNIT_INITIALISMS:
                continue
            if tok.lower() in self._id_keywords or tok.lower() in self._label_words:
                continue
            yield m.start(), m.end(), \
                f"{self._letters_spelled(tok)} {L.cardinal(int(m.group(2)))}"

    # --- unit initialisms after numbers: 1.5 GB, 5,000 mAh ------------------
    def _m_unit_initialism(self, text):
        L = self.lang
        units = "|".join(sorted(_UNIT_INITIALISMS, key=len, reverse=True))
        pat = re.compile(
            rf"(?<![\w.])(-?)\b(\d+(?:,\d{{2,3}})*(?:\.\d+)?)[\s\-]?({units})"
            rf"(?:\s?/\s?([A-Za-z]{{1,4}}))?\b",
            re.IGNORECASE)
        for m in pat.finditer(text):
            unit = m.group(3)
            if unit.upper() not in _UNIT_INITIALISMS:
                continue
            # single-letter units (V, W) only when written in caps â€”
            # a lowercase "5 v" is too ambiguous
            if len(unit) == 1 and not unit.isupper():
                continue
            num = self._number_words(m.group(2))
            if m.group(1):
                num = f"{L.SYM['minus']} {num}"
            uk = re.sub(r"[^A-Za-z]", "", unit).upper()
            word = L.UNIT_WORDS.get(uk) or self._letters_spelled(uk)
            rep = L.join_amount(num, word)
            if m.group(4):
                u2 = m.group(4).upper()
                w2 = L.UNIT_WORDS.get(u2) or self._letters_spelled(u2)
                rep += f" {L.SYM.get('per', 'per')} {w2}"
            yield m.start(), m.end(), rep

    # --- number-hyphen-word: 80-watt, 900-point ------------------------------
    def _m_num_hyphen_word(self, text):
        for m in re.finditer(r"\b(\d+(?:,\d{2,3})*(?:\.\d+)?)-([A-Za-zà¤€-àµ¿]{2,})\b",
                             text):
            word = m.group(2)
            if self._month_index(word):
                continue  # 12-Aug
            if word.upper() in _UNIT_INITIALISMS:
                continue  # unit matcher handles
            num = self._number_words(m.group(1))
            yield m.start(), m.end(), f"{num} {word}"

    # --- decades ------------------------------------------------------------
    def _m_decade(self, text):
        L = self.lang
        for m in re.finditer(r"\b(19|20)(\d)0s\b", text):
            if m.group(1) == "19":
                # 1980s -> "eighty's decade" (spec)
                yield m.start(), m.end(), L.decade(int(m.group(2)) * 10, full=True)
            else:
                # 2020s -> full year words + decade template
                y = int(m.group(0)[:4])
                yield m.start(), m.end(), _fmt(L.DECADE_FULL, tens=L.year(y))
        for m in re.finditer(r"(?<![\w])'?(\d)0s\b", text):
            yield m.start(), m.end(), L.decade(int(m.group(1)) * 10, full=False)

    # --- years ---------------------------------------------------------------
    def _m_year(self, text):
        L = self.lang
        year_words = {w.lower() for w in L.YEAR_WORDS} | \
            {"year", "in", "since", "by", "from", "à¤¸à¤¾à¤²", "à¤µà¤°à¥à¤·"}
        for m in re.finditer(r"\b(1[5-9]\d{2}|20\d{2})\b", text):
            y = int(m.group(0))
            ctx = self._keyword_before(text, m.start(), year_words, window=2)
            # attached native locative suffix right after: 1947à¤®à¥‡à¤‚ / 1947-à®²à¯
            after = text[m.end():m.end() + 20]
            # danda à¥¤/à¥¥ are inside the Devanagari block â€” they are sentence
            # punctuation, not a locative suffix
            suffixed = bool(re.match(r"^-?(?![à¥¤à¥¥])[à¤€-à·¿]", after))
            # following token is/starts with a year word: "1947 à¦¸à¦¾à¦²à§‡",
            # "1983 à¤¸à¤¾à¤²à¥€", "à¥§à¥¯à¥ªà¥­ à¬®à¬¸à¬¿à¬¹à¬¾à¬°à­‡", or Hindi locatives
            next_tok = after.strip().split()[0] if after.strip() else ""
            follower = next_tok in ("à¤®à¥‡à¤‚", "à¤¸à¥‡", "à¤¤à¤•", "àª®àª¾àª‚", "à¤®à¤§à¥à¤¯à¥‡",
                                    "à¨µà¨¿à©±à¨š", "à¨¤à©‹à¨‚") or any(
                next_tok.startswith(w) for w in L.YEAR_WORDS if w)
            if ctx or suffixed or follower:
                yield m.start(), m.end(), L.year(y)

    # --- quoted single letters: reply 'Y' -----------------------------------
    def _m_quoted_letter(self, text):
        for m in re.finditer(r"['\"â€˜â€œ]([A-Za-z])['\"â€™â€]", text):
            yield m.start(), m.end(), self.lang.letter(m.group(1))

    # --- short mixed tokens: 5G, A4, B12, Y2K, S24, A380 ---------------------
    def _m_short_mixed_token(self, text):
        L = self.lang
        pat = re.compile(r"(?<![\d.])\b(?=[A-Za-z0-9]{2,5}\b)"
                         r"(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])"
                         r"[A-Za-z0-9]+\b")
        for m in pat.finditer(text):
            tok = m.group(0)
            if re.fullmatch(r"\d+(?:st|nd|rd|th|s)", tok):
                continue  # ordinals / decades
            out = []
            for run in re.findall(r"[A-Za-z]+|\d+", tok):
                if run.isdigit():
                    out.append(L.cardinal(int(run)))
                else:
                    out.append(self._letters_spelled(run.upper()))
            yield m.start(), m.end(), " ".join(out)

    # --- decimals -------------------------------------------------------------
    def _m_decimal(self, text):
        # trailing (?!\.?\d) instead of (?![\d.]) so a sentence-final full
        # stop after the decimal ("...830.93.") doesn't block the match
        for m in re.finditer(
                r"(?<![\d.])(\d{1,3}(?:,\d{2,3})+|\d+)\.(\d+)(?!\.?\d)", text):
            yield m.start(), m.end(), \
                self.lang.decimal(int(m.group(1).replace(",", "")), m.group(2))

    # --- negative numbers -------------------------------------------------------
    def _m_negative(self, text):
        L = self.lang
        for m in re.finditer(r"(?<![\w.,-])-(\d+(?:\.\d+)?)", text):
            yield m.start(), m.end(), \
                f"{L.SYM['minus']} {self._number_words(m.group(1))}"

    # --- plain integers ----------------------------------------------------------
    def _m_int(self, text):
        L = self.lang
        for m in re.finditer(r"\b\d{1,3}(?:,\d{2,3})+\b|\b\d+\b", text):
            digits = m.group(0).replace(",", "")
            # street numbers read year-style (1422 à¤—à¤¾à¤‚à¤§à¥€ à¤®à¤¾à¤°à¥à¤— -> à¤šà¥Œà¤¦à¤¹ à¤¸à¥Œ à¤¬à¤¾à¤ˆà¤¸)
            after = text[m.end():m.end() + 30].strip().lower()
            first_words = after.split()[:2]
            if (len(digits) == 4 and digits[0] != "0"
                    and any(w.strip(".,") in _STREET_WORDS for w in first_words)):
                yield m.start(), m.end(), L.year(int(digits))
                continue
            if digits.startswith("0") and len(digits) > 1:
                yield m.start(), m.end(), self._digits_spelled(digits, comma=True)
                continue
            yield m.start(), m.end(), L.cardinal(int(digits))

    # ------------------------------------------------------------------
    # Final sweep: leftover meaningful symbols -> words; decorative junk out
    # ------------------------------------------------------------------
    def _final_sweep(self, text):
        S = self.lang.SYM
        text = re.sub(r"<=|â‰¤", " less than or equal to ", text)
        text = re.sub(r">=|â‰¥", " greater than or equal to ", text)
        text = re.sub(r"Â°\s*C\b", f" {S['celsius']} ", text)
        text = re.sub(r"Â°\s*F\b", f" {S['fahrenheit']} ", text)
        text = text.replace("Â°", f" {S['degree']} ")
        rep = {"%": S["percent"], "@": S["at"], "&": S["amp"],
               "=": S["equals"], "+": S["plus"], "Ã—": S["times"],
               "Ã·": S["divide"], "Â±": S["plus_minus"], "âˆš": S["sqrt"],
               "_": S["underscore"], "#": S["hash"]}
        for sym, word in rep.items():
            text = text.replace(sym, f" {word} ")
        text = text.replace("<", f" {S['lt']} ").replace(">", f" {S['gt']} ")
        text = re.sub(r"[\^*]", " ", text)
        # slashes not consumed by URL/fraction/date/code matchers are junk
        text = re.sub(r"\s*/\s*", " ", text)
        # clause colons read as a pause ("à°…à°¡à±à°°à°¸à±: Door" -> "à°…à°¡à±à°°à°¸à±, Door")
        text = re.sub(r"\s*:\s*", ", ", text)
        text = re.sub(r"[â‚¹$â‚¬Â£](?!\s?\d)", " ", text)  # bare currency signs
        text = re.sub(r"[â€“â€”]", " ", text)
        return text

    # ------------------------------------------------------------------
    # Fallback: absolutely no digits may remain
    # ------------------------------------------------------------------
    def _digit_fallback(self, text):
        def repl(m):
            d = m.group(0)
            if len(d) >= 7 or d.startswith("0"):
                return self._digits_spelled(d, comma=True)
            return self.lang.cardinal(int(d))
        return re.sub(r"\d+", repl, text)

    # ------------------------------------------------------------------
    def normalize(self, text):
        validate_text(text, date_order=self.date_order, two_digit_year_base=self.two_digit_year_base, language=self.lang.code)
        # URLs/emails are opaque identifiers. Protect them before prose cleanup
        # can expand Dr/No inside a path or turn // and parentheses into pauses.
        raw = _ascii_digits(text)
        protected, originals, spans = {}, {}, []
        identifiers = (self._m_email, self._m_url)
        if os.environ.get("FR3_TN_URL_SPELL", "0") == "1":
            identifiers += (self._m_handle,)
        for matcher in identifiers:
            for s, e, spoken in matcher(raw):
                if not any(s < b and e > a for a, b, _ in spans):
                    key = chr(0xF0000 + len(protected))
                    protected[key] = spoken
                    originals[key] = raw[s:e]
                    spans.append((s, e, key))
        for s, e, key in sorted(spans, reverse=True):
            raw = raw[:s] + key + raw[e:]
        text = raw
        text = self._preclean(text)
        try:
            matches = self._gather(text)
        except NormalizationError as exc:
            if exc.span is not None:
                start, end = exc.span
                pieces = [text[:start], text[start:end], text[end:]]
                # Parser coordinates refer to precleaned text containing opaque
                # URI placeholders. Restore their source text before retrying.
                for key, original in originals.items():
                    pieces = [piece.replace(key, original) for piece in pieces]
                exc.source = "".join(pieces)
                exc.span = (len(pieces[0]), len(pieces[0]) + len(pieces[1]))
            raise
        # priority asc, then leftmost, then longest
        matches.sort(key=lambda t: (t[0], t[1], -(t[2] - t[1])))
        taken = []
        accepted = []
        for prio, s, e, rep in matches:
            if any(not (e <= ts or s >= te) for ts, te in taken):
                continue
            taken.append((s, e))
            accepted.append((s, e, rep))
        accepted.sort(key=lambda t: t[0])
        out = []
        pos = 0
        for s, e, rep in accepted:
            out.append(text[pos:s])
            pos = e
            # native postposition right after the span (spaced OR attached:
            # "à°—à°‚à°Ÿà°²à± à°•à°¿" / "6:30à°•à°¿") -> language attaches it with sandhi
            while True:
                nxt = re.match(r"[ \t]*-?([^\s\d,à¥¤.!?;:()\-â€”â€“]{1,14})",
                               text[pos:])
                if not nxt:
                    break
                if text[pos + nxt.end(): pos + nxt.end() + 1] == ".":
                    break  # abbreviation like à°•à°¿.à°®à±€ â€” not a postposition
                merged = self.lang.attach_postposition(rep, nxt.group(1))
                if merged is None:
                    break
                rep = merged
                pos += nxt.end()
            out.append(rep)
        out.append(text[pos:])
        result = "".join(out)
        result = self._final_sweep(result)
        result = self._digit_fallback(result)
        for key, spoken in protected.items():
            result = result.replace(key, spoken)
        # A foreign cardinal must never be glued into an Indic suffix (fiveà°¨).
        result = re.sub(r"(?<=[A-Za-z])(?=[\u0900-\u0DFF])", " ", result)
        # tidy whitespace / punctuation spacing
        result = re.sub(r"\s+([,à¥¤.!?;])", r"\1", result)
        result = re.sub(r",\s*([à¥¤.!?;,])", r"\1", result)
        result = re.sub(r",(?=\S)", ", ", result)  # space after every comma
        result = re.sub(r"\s{2,}", " ", result)
        result = re.sub(r"\s+â€™", "â€™", result)
        result = re.sub(r"[,\s]+$", ".", result) if result.rstrip().endswith(",") else result
        if self.lang.code == "hi":
            result = result.translate(_HINDI_NUKTA_FOLD)
        return result.strip()




# ---------------------------------------------------------------------------
# English table (the only language table)
# ---------------------------------------------------------------------------

_ONES_0_19 = ["zero", "one", "two", "three", "four", "five", "six", "seven",
              "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
              "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _two_digit(n):
    """0..99 -> 'zero' .. 'ninety nine'. No hyphen: the model reads words."""
    if n < 20:
        return _ONES_0_19[n]
    t, o = divmod(n, 10)
    return _TENS[t] if not o else f"{_TENS[t]} {_ONES_0_19[o]}"


_ORDINAL_IRREGULAR = {1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth",
                      9: "ninth", 12: "twelfth"}


def _ordinal(n):
    if n in _ORDINAL_IRREGULAR:
        return _ORDINAL_IRREGULAR[n]
    w = _two_digit(n) if n < 100 else None
    if w is None:
        return None
    # twenty-first, thirty-second ... take the irregular form on the last digit
    tail = n % 10
    if n >= 20 and tail in _ORDINAL_IRREGULAR:
        return f"{_TENS[n // 10]} {_ORDINAL_IRREGULAR[tail]}"
    if w.endswith("y"):                 # twenty -> twentieth
        return w[:-1] + "ieth"
    return w + "th"


class English(LangBase):
    code = "en"
    name = "English"
    native_digits = "0123456789"        # English uses ASCII digits

    ONES = [_two_digit(i) for i in range(100)]
    HUNDRED = "hundred"
    THOUSAND = "thousand"
    LAKH = "lakh"
    CRORE = "crore"

    # Letter names are already English, so each maps to itself.
    LETTERS = {chr(65 + i): chr(65 + i) for i in range(26)}

    MONTHS = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]

    SYM = {
        "rupee": "rupees", "dollar": "dollars", "euro": "euros",
        "pound": "pounds", "cent": "cents",
        "percent": "percent",
        "point": "point",
        "minus": "minus", "plus": "plus", "equals": "equals",
        "times": "times", "divide": "divided by",
        "squared": "squared", "sqrt": "square root of",
        "lt": "less than", "gt": "greater than",
        "plus_minus": "plus or minus",
        "at": "at", "underscore": "underscore", "dot": "dot",
        "slash": "slash", "hash": "hash", "amp": "and",
        "half": "half", "quarter": "quarter", "three_quarters": "three quarters",
        "degree": "degrees", "celsius": "celsius", "fahrenheit": "fahrenheit",
        "minute": "minutes",
    }

    TIME_ON_HOUR = "{h} o'clock"
    TIME_HM = "{h} {m}"                 # "ten thirty" â€” how a clock is read aloud
    RANGE = "{a} to {b}"
    RANGE_TO = "{a} to {b}"
    RATIO = "{a} to {b}"
    POWER = "{base} to the power {exp}"
    FRACTION = "{num} over {den}"
    DECADE_SHORT = "{tens}s"
    DECADE_FULL = "{tens}s"

    PERIODS = {"morning": "in the morning", "afternoon": "in the afternoon",
               "evening": "in the evening", "night": "at night"}

    # ---- vocabulary the engine must recognise in the INPUT ----------------
    TIME_MARKERS = ["o'clock", "oclock"]
    MARKER_SUFFIXES = {}
    PERIOD_DETECT = {"morning": "morning", "afternoon": "afternoon",
                     "evening": "evening", "night": "night",
                     "am": "morning", "a.m.": "morning",
                     "pm": "afternoon", "p.m.": "afternoon"}
    YEAR_WORDS = ["year", "years"]
    # Deliberately NOT "order"/"invoice"/"ticket" on their own: "your order is 250
    # rupees" is a price, and listing them made it read "two, five, zero". They only
    # mark an identifier in the explicit "order number ..." form, which "number" catches.
    # One normalizer serves every language, so an identifier must be recognised
    # whatever script the sentence around it is written in. Without the native
    # words, "à¤‘à¤°à¥à¤¡à¤° à¤¨à¤‚à¤¬à¤° 47291" reads as a cardinal instead of digit-by-digit.
    ID_KEYWORDS = ["number", "pin", "otp", "pnr", "imei", "ifsc", "account", "à¤¨à¤‚à¤¬à¤°", "à¤¨à¤®à¥à¤¬à¤°",
        "à¤–à¤¾à¤¤à¤¾", "à¤–à¤¾à¤¤à¥‡", "à¤•à¥‹à¤¡", "à¤‘à¤°à¥à¤¡à¤°", "à¤«à¤¼à¥‹à¤¨", "à¤«à¥‹à¤¨", "à¤®à¥‹à¤¬à¤¾à¤‡à¤²", "à°¨à°‚à°¬à°°à±", "à°¨à°‚à°¬à°°à±", "à°¨à±†à°‚à°¬à°°à±",
        "à°¨à±†à°‚à°¬à°°à±", "à°–à°¾à°¤à°¾", "à°«à±‹à°¨à±", "à°®à±Šà°¬à±ˆà°²à±", "à°•à±‹à°¡à±", "à°†à°°à±à°¡à°°à±", "à°Žà°•à±à°¸à±à°ªà±à°°à±†à°¸à±", "express",
        "à°…à°•à±Œà°‚à°Ÿà±", "à°Ÿà°¿à°•à±†à°Ÿà±", "à°°à°¿à°œà°¿à°¸à±à°Ÿà±à°°à±‡à°·à°¨à±", "à°¯à±‚à°œà°°à±", "à°µà°¾à°¯à°¿à°¦à°¾", "à°…à°‚à°•à±†", "à°²à°¾à°µà°¾à°¦à±‡à°µà±€", "à°ªà°¿à°¨à±",
        "à®Žà®£à¯", "à®¨à®®à¯à®ªà®°à¯", "à®•à®£à®•à¯à®•à¯", "à®¤à¯Šà®²à¯ˆà®ªà¯‡à®šà®¿", "à®ªà¯‹à®©à¯", "à®®à¯Šà®ªà¯ˆà®²à¯", "à®†à®°à¯à®Ÿà®°à¯", "à®“à®Ÿà®¿à®ªà®¿", "à®ªà®¿à®©à¯",
        "à®à®Ÿà®¿", "à®•à®Ÿà®µà¯à®šà¯à®šà¯Šà®²à¯", "à®°à®¯à®¿à®²à¯", "à®Žà®•à¯à®¸à¯à®ªà®¿à®°à®¸à¯", "à®ªà®¿à®Žà®©à¯à®†à®°à¯", "à®•à®¾à®°à¯à®Ÿà¯", "à®•à¯à®±à®¿à®¯à¯€à®Ÿà¯",
        "à®Ÿà®¿à®•à¯à®•à¯†à®Ÿà¯", "à®•à®£à®•à¯à®•à®¿à®²à¯", "à®¯à¯‚à®šà®°à¯", "à®¤à®µà®£à¯ˆ", "à®‡à®²à®•à¯à®•à®®à¯", "à®ªà®°à®¿à®µà®°à¯à®¤à¯à®¤à®©à¯ˆ", "à²¨à²‚à²¬à²°à³", "à²¸à²‚à²–à³à²¯à³†",
        "à²–à²¾à²¤à³†", "à²•à³‹à²¡à³", "à²†à²°à³à²¡à²°à³", "à²«à³‹à²¨à³", "à²®à³Šà²¬à³ˆà²²à³", "à²à²¡à²¿", "à²ªà²¿à²¨à³", "à²ªà²¾à²¸à³â€Œà²µà²°à³à²¡à³", "à²°à³ˆà²²à³",
        "à²Žà²•à³à²¸à³â€Œà²ªà³à²°à³†à²¸à³", "à²ªà²¿à²Žà²¨à³â€Œà²†à²°à³", "à²•à²¾à²°à³à²¡à³", "à²°à³†à²«à²°à³†à²¨à³à²¸à³", "à²Ÿà³à²°à²¾à²¨à³à²¸à²¾à²•à³à²·à²¨à³", "à²Ÿà²¿à²•à³†à²Ÿà³", "à²“à²Ÿà²¿à²ªà²¿",
        "à²…à²•à³Œà²‚à²Ÿà³", "à²…à²‚à²•à²¿", "à²¬à²³à²•à³†à²¦à²¾à²°", "à²•à²‚à²¤à³", "à²µà²¹à²¿à²µà²¾à²Ÿà³", "à´¨à´®àµà´ªàµ¼", "à´¨à´®àµà´ªà´°àµ", "à´¨à´®àµà´ªà´±à´¿àµ½",
        "à´…à´•àµà´•àµ—à´£àµà´Ÿàµ", "à´•àµ‹à´¡àµ", "à´“àµ¼à´¡àµ¼", "à´«àµ‹àµº", "à´®àµŠà´¬àµˆàµ½", "à´’à´Ÿà´¿à´ªà´¿", "à´ªà´¿àµ»", "à´à´¡à´¿", "à´ªà´¾à´¸àµâ€Œà´µàµ‡à´¡àµ",
        "à´Ÿàµà´°àµ†à´¯à´¿àµ»", "à´Žà´•àµà´¸àµà´ªàµà´°à´¸àµ", "à´ªà´¿à´Žàµ»à´†àµ¼", "à´•à´¾àµ¼à´¡àµ", "à´±à´«à´±àµ»à´¸àµ", "à´Ÿàµà´°à´¾àµ»à´¸à´¾à´•àµà´·àµ»", "à´Ÿà´¿à´•àµà´•à´±àµà´±àµ",
        "à´…à´•àµà´•àµ—à´£àµà´Ÿà´¿àµ½", "à´¨à´®àµà´ªà´±à´¿à´²àµ‡à´•àµà´•àµ", "à´¯àµ‚à´¸àµ¼", "à´¤à´µà´£", "à´…à´•àµà´•à´‚", "à´‡à´Ÿà´ªà´¾à´Ÿàµ", "à¤•à¥à¤°à¤®à¤¾à¤‚à¤•", "à¤–à¤¾à¤¤à¥à¤¯à¤¾à¤šà¤¾",
        "à¤–à¤¾à¤¤à¥à¤¯à¤¾à¤µà¤°", "à¤®à¥‹à¤¬à¤¾à¤ˆà¤²", "à¤…à¤•à¤¾à¤‰à¤‚à¤Ÿ", "à¤…à¤•à¤¾à¤‰à¤‚à¤Ÿà¤¸à¤¾à¤ à¥€", "à¤…à¤•à¤¾à¤‰à¤‚à¤Ÿà¤®à¤§à¥‚à¤¨", "à¤–à¤¾à¤¤à¥à¤¯à¤¾à¤¤à¥‚à¤¨", "à¤–à¤¾à¤¤à¥à¤¯à¤¾à¤¤",
        "à¤–à¤¾à¤¤à¥à¤¯à¤¾à¤šà¥€", "à¤ªà¤¿à¤¨", "à¤†à¤¯à¤¡à¥€", "à¤ªà¤¾à¤¸à¤µà¤°à¥à¤¡", "à¤Ÿà¥à¤°à¥‡à¤¨", "à¤à¤•à¥à¤¸à¤ªà¥à¤°à¥‡à¤¸", "à¤ªà¥€à¤à¤¨à¤†à¤°", "à¤•à¤¾à¤°à¥à¤¡",
        "à¤°à¥‡à¤«à¤°à¤¨à¥à¤¸", "à¤Ÿà¥à¤°à¤¾à¤‚à¤à¥…à¤•à¥à¤¶à¤¨", "à¤¤à¤¿à¤•à¥€à¤Ÿ", "à¤“à¤Ÿà¥€à¤ªà¥€", "à¤—à¤¾à¤¡à¥€", "à¤¯à¥à¤œà¤°", "à¤¯à¥‚à¤œà¤°", "à¤¹à¤ªà¥à¤¤à¤¾",
        "à¤¹à¤ªà¥à¤¤à¥à¤¯à¤¾à¤šà¤¾", "à¤…à¤‚à¤•", "à¤µà¥à¤¯à¤µà¤¹à¤¾à¤°", "à¤Ÿà¥à¤°à¤¾à¤¨à¥à¤à¥…à¤•à¥à¤¶à¤¨", "à¤•à¥à¤°", "à¤•à¥à¤°à¤‚", "à¤šà¥‡à¤•", "à¤§à¤¨à¤¾à¤¦à¥‡à¤¶", "à¦¨à¦®à§à¦¬à¦°",
        "à¦¨à¦¾à¦®à§à¦¬à¦¾à¦°", "à¦…à§à¦¯à¦¾à¦•à¦¾à¦‰à¦¨à§à¦Ÿ", "à¦à¦•à¦¾à¦‰à¦¨à§à¦Ÿ", "à¦•à§‹à¦¡", "à¦…à¦°à§à¦¡à¦¾à¦°", "à¦«à§‹à¦¨", "à¦®à§‹à¦¬à¦¾à¦‡à¦²", "à¦“à¦Ÿà¦¿à¦ªà¦¿", "à¦ªà¦¿à¦¨",
        "à¦†à¦‡à¦¡à¦¿", "à¦ªà¦¾à¦¸à¦“à¦¯à¦¼à¦¾à¦°à§à¦¡", "à¦Ÿà§à¦°à§‡à¦¨", "à¦à¦•à§à¦¸à¦ªà§à¦°à§‡à¦¸", "à¦ªà¦¿à¦à¦¨à¦†à¦°", "à¦•à¦¾à¦°à§à¦¡", "à¦°à§‡à¦«à¦¾à¦°à§‡à¦¨à§à¦¸",
        "à¦Ÿà§à¦°à¦¾à¦¨à¦œà§‡à¦•à¦¶à¦¨", "à¦Ÿà¦¿à¦•à¦¿à¦Ÿ", "à¦…à§à¦¯à¦¾à¦•à¦¾à¦‰à¦¨à§à¦Ÿà§‡à¦°", "à¦¨à¦®à§à¦¬à¦°à§‡", "à¦†à¦‡à¦¡à¦¿à¦¤à§‡", "à¦‡à¦‰à¦œà¦¾à¦°", "à¦‡à¦¨à¦¸à§à¦Ÿà¦²à¦®à§‡à¦¨à§à¦Ÿ",
        "à¦•à¦¿à¦¸à§à¦¤à¦¿", "à¦¡à¦¿à¦œà¦¿à¦Ÿ", "à¦…à¦™à§à¦•", "à¦²à§‡à¦¨à¦¦à§‡à¦¨", "àª¨àª‚àª¬àª°", "àª¨àª®à«àª¬àª°", "àª–àª¾àª¤àª¾", "àª–àª¾àª¤à«àª‚", "àª–àª¾àª¤àª¾àª‚", "àª•à«‹àª¡",
        "àª“àª°à«àª¡àª°", "àª«à«‹àª¨", "àª®à«‹àª¬àª¾àª‡àª²", "àª®à«‹àª¬àª¾àªˆàª²", "àªàª•àª¾àª‰àª¨à«àªŸ", "àªàª•àª¾àª‰àª¨à«àªŸàª®àª¾àª‚", "àªàª•àª¾àª‰àª¨à«àªŸàª¨à«‹", "àªªàª¿àª¨",
        "àª†àªˆàª¡à«€", "àªªàª¾àª¸àªµàª°à«àª¡", "àªŸà«àª°à«‡àª¨", "àªàª•à«àª¸àªªà«àª°à«‡àª¸", "àªªà«€àªàª¨àª†àª°", "àª•àª¾àª°à«àª¡", "àª°à«‡àª«àª°àª¨à«àª¸", "àªŸà«àª°àª¾àª¨à«àªà«‡àª•à«àª¶àª¨",
        "àªŸàª¿àª•àª¿àªŸ", "àª“àªŸà«€àªªà«€", "àª¨àª‚àª¬àª°àª®àª¾àª‚", "àª¯à«àªàª°", "àª¹àªªà«àª¤à«‹", "àª…àª‚àª•", "àª¡àª¿àªœàª¿àªŸ", "à¨¨à©°à¨¬à¨°", "à¨–à¨¾à¨¤à¨¾", "à¨–à¨¾à¨¤à©‡",
        "à¨•à©‹à¨¡", "à¨†à¨°à¨¡à¨°", "à¨«à¨¼à©‹à¨¨", "à¨«à©‹à¨¨", "à¨®à©‹à¨¬à¨¾à¨ˆà¨²", "à¨®à©‹à¨¬à¨¾à¨‡à¨²", "à¨…à¨•à¨¾à¨Šà¨‚à¨Ÿ", "à¨…à¨•à¨¾à¨Šà¨‚à¨Ÿà¨µà¨¿à©±à¨š", "à¨ªà¨¿à©°à¨¨",
        "à¨†à¨ˆà¨¡à©€", "à¨ªà¨¾à¨¸à¨µà¨°à¨¡", "à¨Ÿà©à¨°à©‡à¨¨", "à¨à¨•à¨¸à¨ªà©à¨°à©ˆà¨¸", "à¨ªà©€à¨à¨¨à¨†à¨°", "à¨•à¨¾à¨°à¨¡", "à¨°à©ˆà¨«à¨°à©ˆà¨‚à¨¸", "à¨Ÿà¨¿à¨•à¨Ÿ", "à¨“à¨Ÿà©€à¨ªà©€",
        "à¨¯à©‚à¨œà¨¼à¨°", "à¨•à¨¿à¨¸à¨¼à¨¤", "à¨…à©°à¨•", "à¨²à©ˆà¨£-à¨¦à©‡à¨£", "à¬¨à¬®à­à¬¬à¬°", "à¬¨à¬®à­à¬¬à¬°à­", "à¬–à¬¾à¬¤à¬¾", "à¬†à¬•à¬¾à¬‰à¬£à­à¬Ÿ", "à¬•à­‹à¬¡à­",
        "à¬…à¬°à­à¬¡à¬°", "à¬«à­‹à¬¨à­", "à¬«à­‹à¬¨", "à¬®à­‹à¬¬à¬¾à¬‡à¬²à­", "à¬®à­‹à¬¬à¬¾à¬‡à¬²", "à¬“à¬Ÿà¬¿à¬ªà¬¿", "à¬ªà¬¿à¬¨à­", "à¬†à¬‡à¬¡à¬¿", "à¬ªà¬¾à¬¸à­±à¬¾à¬°à­à¬¡",
        "à¬à¬•à­à¬¸à¬ªà­à¬°à­‡à¬¸", "à¬ªà¬¿à¬à¬¨à­â€Œà¬†à¬°à­", "à¬•à¬¾à¬°à­à¬¡", "à¬°à­‡à¬«à¬°à­‡à¬¨à­à¬¸", "à¬Ÿà¬¿à¬•à­‡à¬Ÿ", "à¬†à¬•à¬¾à¬‰à¬£à­à¬Ÿà¬°à­", "à¬Ÿà­à¬°à­‡à¬¨à­", "à­Ÿà­à¬œà¬°",
        "à¬•à¬¿à¬¸à­à¬¤à¬¿", "à¬…à¬™à­à¬•", "à¬²à¬¾à¬£à¬¦à­‡à¬£"]
    ID_AFTER_WORDS = ["number", "code", "à¤–à¤¤à¥à¤®", "à¤¡à¤¾à¤¯à¤²", "à¤•à¥‰à¤²", "à¤¸à¤‚à¤ªà¤°à¥à¤•", "à°Žà°‚à°¡à±", "à°®à±à°—à°¿à°¸à±‡",
        "à°®à±à°—à±à°¸à±à°¤à±à°‚à°¦à°¿", "à°šà°¿à°µà°°", "à°šà°¿à°µà°°à°¿", "à°…à°‚à°•à±†à°²à°¤à±‹", "à°¡à°¯à°²à±", "à°ªà°¿à°¨à±", "à°ªà°¿à°¨à±â€Œà°•à±‹à°¡à±", "pin",
        "pincode", "à°•à°¾à°²à±", "à°¸à°‚à°ªà±à°°à°¦à°¿à°‚à°šà°‚à°¡à°¿", "à®Ÿà®¯à®²à¯", "à®…à®´à¯ˆ", "à®¤à¯Šà®Ÿà®°à¯à®ªà¯", "à²¡à²¯à²²à³", "à²•à²°à³†", "à²¸à²‚à²ªà²°à³à²•",
        "à¦¡à¦¾à¦¯à¦¼à¦¾à¦²", "à¦•à¦²", "à¦¯à§‹à¦—à¦¾à¦¯à§‹à¦—", "àª¡àª¾àª¯àª²", "àª•à«‰àª²", "àª¸àª‚àªªàª°à«àª•", "à¨¡à¨¾à¨‡à¨²", "à¨•à¨¾à¨²", "à¨¸à©°à¨ªà¨°à¨•", "à¬¡à¬¾à¬à¬²",
        "à¬•à¬²", "à¬¯à­‹à¬—à¬¾à¬¯à­‹à¬—"]
    LABEL_WORDS = ["room", "flat", "floor", "gate", "platform", "seat",
                   "section", "chapter", "page", "block", "phase"]
    MINUTE_WORDS = ["minute", "minutes", "min", "mins"]
    SCALE_WORDS = {"crore": "crore", "crores": "crore", "cr": "crore",
                   "lakh": "lakh", "lakhs": "lakh", "lac": "lakh",
                   "thousand": "thousand", "thousands": "thousand", "k": "thousand"}
    CURRENCY_WORDS = {"rupee": "rupee", "rupees": "rupee", "rs": "rupee",
                      "inr": "rupee", "dollar": "dollar", "dollars": "dollar",
                      "usd": "dollar", "euro": "euro", "euros": "euro",
                      "pound": "pound", "pounds": "pound",
                      "paisa": "cent", "paise": "cent",
                      "cent": "cent", "cents": "cent"}

    # Unit initialisms spoken as full words instead of letters. The engine
    # falls back to letter-spelling ("K G") when a key is absent, which is
    # right for GB/MB but wrong for the metric units customers actually say.
    UNIT_WORDS = {
        "KG": "kilograms", "KM": "kilometers", "CM": "centimeters",
        "MM": "millimeters", "GM": "grams", "MG": "milligrams",
        "ML": "millilitres", "MCG": "micrograms",
        "KMPH": "kilometers per hour", "MPH": "miles per hour",
        "KWH": "kilowatt hours", "RPM": "R P M",
    }

    # Written-only abbreviations expanded wherever they appear. Case-insensitive,
    # guarded by word boundaries, so "Drew" and "MIN-22" are untouched.
    ABBREV_UNITS = {
        "MINS": "minutes", "MIN": "minutes", "HRS": "hours", "SECS": "seconds",
        "DR": "doctor", "MR": "mister", "MRS": "missus", "MS": "miss",
        "PROF": "professor", "LTD": "limited", "PVT": "private",
        "A/C": "account", "APPROX": "approximately", "ETA": "E T A",
    }

    # "No. 12" -> "number twelve"
    NUMBER_WORD = "number"

    # English transcribes the am/pm that was written; it never invents a period.
    # "6.45 pm" -> "six forty five p m", NOT "six forty five in the evening".
    AMPM_WORDS = {"am": "a m", "pm": "p m"}

    ORDINAL_EXCEPTIONS = {n: _ordinal(n) for n in range(1, 100) if _ordinal(n)}
    ORDINAL_SUFFIX = "th"
    ORDINAL_INPUT_SUFFIXES = ["st", "nd", "rd", "th"]


    def time_phrase(self, h, m, period_key=None):
        """English puts the period AFTER the clock time â€” "five in the evening", not
        the base class's Indo-Aryan "in the evening five" (à¤¶à¤¾à¤® à¤ªà¤¾à¤à¤š à¤¬à¤œà¥‡)."""
        if h == 0:
            core = "zero " + ("hundred" if m == 0 else self.two_digit(m))
        else:
            core = super().time_phrase(h, m, None)
        return f"{core} {self.PERIODS[period_key]}" if period_key else core

    def _ordinal_compose(self, n, src_suffix=""):
        if n < 100:
            return "zeroth" if n == 0 else _ordinal(n)
        for scale, word in ((10**7, self.CRORE), (10**5, self.LAKH),
                            (1000, self.THOUSAND), (100, self.HUNDRED)):
            if n >= scale:
                count, rest = divmod(n, scale)
                head = f"{self.cardinal(count)} {word}"
                return f"{head} {self.ordinal(rest)}" if rest else head + "th"


LANG = English()
_NORMALIZER = Normalizer(LANG)


@lru_cache(maxsize=66)
def _engine(language, date_order, two_digit_year_base):
    return Normalizer(language_table(language), date_order=date_order, two_digit_year_base=two_digit_year_base)


def normalize(text, language="en-IN", *, date_order="DMY", two_digit_year_base=None):
    """Locale-specific numbers and symbols; prose is preserved, never translated.

    Omitted language retains Indian English for compatibility. Use an explicit
    language/locale for Indic speech. language="auto" selects majority-script
    inference with a guard against introducing foreign-script expansions.
    """
    auto = isinstance(language, str) and language == "auto"
    original = text
    if auto:
        if __package__:
            from .script_guard import detect_language, guard_output
        else:
            from script_guard import detect_language, guard_output
        language = detect_language(text) if isinstance(text, str) else "en-IN"
    code = resolve_language(language)
    # Validate before accessing the cache, including unhashable option values.
    warning = None
    while True:
        try:
            validate_text(text, date_order=date_order, two_digit_year_base=two_digit_year_base, language=code)
            result = _engine(code, date_order, two_digit_year_base).normalize(text)
            break
        except NormalizationError as exc:
            # Request type/options are not text tokens and cannot be rendered.
            if os.environ.get("FR3_TN_STRICT", "0") == "1" or exc.span is None:
                raise
            warning = warning or exc.code
            source = exc.source if exc.source is not None else text
            start, end = exc.span
            token = source[start:end]
            locale = language_table(code)
            symbols = {"/": "slash", ".": "dot", "-": "minus", "+": "plus",
                       "@": "at", "_": "underscore", "â‚¹": "rupee", "$": "dollar",
                       "%": "percent"}
            words = []
            for char in token:
                if char.isdecimal():
                    words.append(" " + locale.ONES[unicodedata.decimal(char)] + " ")
                elif char in symbols:
                    words.append(" " + locale.SYM[symbols[char]] + " ")
                elif char in ",:;<>{}" or unicodedata.category(char) in ("Cc", "Co"):
                    words.append(" ")
                else:
                    words.append(char)
            replacement = "".join(words)
            if replacement == token:
                raise  # no progress: do not loop on a new, unhandled error
            text = source[:start] + " " + replacement + " " + source[end:]
    if warning:
        # One privacy-safe observation per normalization call, never caller text.
        try:
            import obs
            obs.event("normalization_fallback", level="warn", code=warning)
        except ImportError:
            print("[obs] " + json.dumps({"event": "normalization_fallback", "level": "warn", "code": warning}), flush=True)
    return guard_output(original, result, language_table(code)) if auto else result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Plain text; otherwise read stdin.")
    parser.add_argument("--date-order", choices=("DMY", "MDY"), default="DMY")
    parser.add_argument("--language", default="en-IN", help="Language code/name, including hi, te and en-IN.")
    parser.add_argument("--two-digit-year-base", type=int, choices=(1900, 2000))
    args = parser.parse_args()
    try:
        print(normalize(args.text if args.text is not None else sys.stdin.read(), language=args.language,
                        date_order=args.date_order, two_digit_year_base=args.two_digit_year_base))
    except NormalizationError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        sys.exit(2)
'''),
    'mithi_tts.text_norm.tn.script_guard': (False, r'''"""Deterministic, CPU-only locale inference and output script containment.

Count letters/marks, not shared danda punctuation or decimal digits. Devanagari
cannot distinguish Hindi/Marathi: auto uses Hindi; explicit Marathi is preferred.
"""
import unicodedata
from collections import Counter

BLOCKS = ((0x0900, 0x097F, 'hi'), (0x0980, 0x09FF, 'bn'),
          (0x0A00, 0x0A7F, 'pa'), (0x0A80, 0x0AFF, 'gu'),
          (0x0B00, 0x0B7F, 'or'), (0x0B80, 0x0BFF, 'ta'),
          (0x0C00, 0x0C7F, 'te'), (0x0C80, 0x0CFF, 'kn'),
          (0x0D00, 0x0D7F, 'ml'))


def script_counts(text):
    counts = Counter()
    for c in text:
        if unicodedata.category(c)[0] not in 'LM':
            continue
        if unicodedata.name(c, '').startswith('LATIN'):
            counts['en-IN'] += 1
            continue
        for low, high, code in BLOCKS:
            if low <= ord(c) <= high:
                counts[code] += 1
                break
    return counts


def detect_language(text):
    counts = script_counts(text)
    code = max(counts, key=counts.get) if counts else 'en-IN'
    if code == 'bn' and any(c in text for c in 'à§°à§±'):
        return 'as'
    return code


def guard_output(original, result, locale):
    allowed = set(script_counts(original))
    # A numeric/symbol-only input supplies no script; use the default locale.
    if not allowed or set(script_counts(result)) <= allowed:
        return result
    # Fail conservatively: preserve prose and read the original digits in the
    # chosen input script; never return a generated foreign-script expansion.
    if __package__:
        from .normalize import language_table
    else:
        from normalize import language_table
    locale = language_table(detect_language(original))
    result = ''.join(' ' + locale.digit(unicodedata.decimal(c)) + ' '
                     if c.isdecimal() else c for c in original)
    return ' '.join(result.split()) if set(script_counts(result)) <= allowed else original
'''),
    'mithi_tts.text_norm.tn.text_pipeline': (False, r'''"""CPU-only request preparation shared by HTTP and WebSocket serving.

No model imports, logging of caller text, or silent raw-text fallback.
"""
if __package__:
    from .normalize import LANGUAGE_NAMES, NormalizationError, validate_text, resolve_language
    from .script_guard import detect_language
else:
    from normalize import LANGUAGE_NAMES, NormalizationError, validate_text, resolve_language
    from script_guard import detect_language

NORMALIZATION_LANGUAGES = tuple(LANGUAGE_NAMES)


class PipelineError(Exception):
    def __init__(self, code, message, status=422):
        self.code, self.status = code, status
        super().__init__(message)

    def payload(self):
        return {"error": str(self), "code": self.code}


def prepare_text(text, *, normalizer, splitter, post_process, normalize_enabled=True,
                 chunk_enabled=True, date_order="DMY", two_digit_year_base=None, max_chunk_chars=None,
                 normalization_language="en-IN"):
    try:
        if not normalize_enabled:
            validate_text(text, date_order=date_order, two_digit_year_base=two_digit_year_base, language=detect_language(text) if normalization_language is None and isinstance(text, str) else normalization_language)
        if not isinstance(text, str):
            raise NormalizationError("invalid_text", "text must be a string.")
        t = text.strip()
        if not t:
            return []
        if normalize_enabled:
            if normalizer is None:
                raise PipelineError("normalizer_unavailable", "Text normalizer unavailable.", 503)
            t = normalizer(t, language="auto" if normalization_language is None else normalization_language, date_order=date_order, two_digit_year_base=two_digit_year_base)
            if not isinstance(t, str) or not any(c.isalnum() for c in t):
                raise PipelineError("empty_normalized_text", "No speakable text remains.")
        if chunk_enabled:
            if splitter is None or post_process is None:
                raise PipelineError("chunker_unavailable", "Text chunker unavailable.", 503)
            t = post_process(t)
            parts = splitter(t)
            if not parts or any(not isinstance(p, str) or not p.strip() for p in parts):
                raise PipelineError("invalid_chunks", "Text chunker returned invalid chunks.", 503)
            # Current chunker changes whitespace only. Detect dropped/reordered text
            # before audio starts; do not send a too-long whole-text fallback.
            if " ".join(" ".join(parts).split()) != " ".join(t.split()):
                raise PipelineError("chunk_content_mismatch", "Text chunker changed content.", 503)
            if max_chunk_chars and any(len(p) > max_chunk_chars for p in parts):
                raise PipelineError("unsplittable_text", "A text token exceeds the safe chunk size; shorten it.")
            return parts
        return [t]
    except NormalizationError as exc:
        raise PipelineError(exc.code, str(exc)) from None
    except PipelineError:
        raise
    except Exception:
        raise PipelineError("text_pipeline_failed", "Text preparation failed; no audio was generated.", 503) from None


def resolve_conditioning(settings, supported):
    """Never invent an untrained language tag. Legacy omission remains Hindi.

    `language`/`lang` requests a model tag unless `conditioning_language` is
    explicitly supplied. The latter is for a checkpoint trained with a tag that
    differs from the text language, and requires deployment-owner confirmation.
    """
    value = settings.get("conditioning_language") or settings.get("language") or settings.get("lang") or "Hindi"
    aliases = {"en": "English", "en-in": "English", "en-us": "English",
               "hi": "Hindi", "hi-in": "Hindi", "indian english": "English", "indian-english": "English"}
    if not isinstance(value, str):
        raise PipelineError("invalid_language", "language must be a string.")
    by_name = {x.lower(): x for x in supported}
    value = aliases.get(value.lower(), by_name.get(value.lower(), value))
    if value not in supported:
        raise PipelineError("unsupported_conditioning_language",
                            "No supported model tag for this language. Confirm the checkpoint conditioning_language.")
    return value


def resolve_normalization(settings):
    """Normalization locale and model conditioning are deliberately separate.

    Explicit normalization_language wins, then language/lang; omission preserves
    Indian English. Never guess Hindi versus Marathi from Devanagari text.
    """
    value = next((settings[key] for key in ("normalization_language", "language", "lang")
                  if key in settings), "en-IN")
    try:
        return resolve_language(value)
    except NormalizationError as exc:
        raise PipelineError(exc.code, str(exc)) from None


def merge_settings(current, incoming, sticky):
    """Snapshot per queued utterance; newer aliases replace older aliases."""
    result = dict(current)
    # A language change must not inherit a previous explicit override.
    if "language" in incoming or "lang" in incoming:
        for key in ("conditioning_language", "normalization_language"):
            if key not in incoming:
                result.pop(key, None)
    for a, b in (("voice_id", "voice"), ("language", "lang")):
        if a in incoming or b in incoming:
            result.pop(a, None)
            result.pop(b, None)
    result.update({k: incoming[k] for k in sticky if k in incoming})
    return result


_REQUEST_LANGUAGE_KEYS = frozenset(("language", "lang", "normalization_language", "conditioning_language"))


def retain_session_settings(current, incoming, sticky):
    """Text-frame locales belong to their context, not the next customer turn.

    start/config frames still use merge_settings directly and establish explicit
    session defaults. Other sticky options retain the existing protocol behavior.
    """
    return merge_settings(current, {k: v for k, v in incoming.items()
                                   if k not in _REQUEST_LANGUAGE_KEYS}, sticky)
'''),
    'mithi_tts.text_norm.tn.tts_chunker': (False, r'''"""Prosody-aware, Indic-aware, emotion-tag-safe chunker for TTS.

Python port of the TypeScript chunker (studio gateway, itself a port of
indic_chunker.py on branch local-chunker-normguard) that passes the I1-I6
invariant harness. Replaces the earlier sentence-only chunker, which had two
defects:
  * it split on any '.', '!' or '?', tearing "8.4", "example.com", "â‚¹2,500"
    and quoted '"à¤†à¤“à¤—à¥‡?"' apart;
  * it split ONLY on sentence punctuation, so a long comma-joined sentence
    became one oversized chunk that silently overran max_model_len and
    truncated mid-speech.

Splits normalized text preferring, in priority order:
  1. sentence boundaries  ( . ! ? â€¦ à¥¤ à¥¥ â€” ONLY token-final, i.e. followed by
     whitespace, so decimals / URLs / currency / quoted '?' stay intact )
  2. clause boundaries    ( ; : , â€” â€“  and pan-Indic connector words )
  3. word boundaries      ( never splits a word / Devanagari cluster )

Emotion tags ([laughs], [clears throat], â€¦) are atomic and glued to the word
they condition, so a boundary can never strand a tag into the wrong clip.

BUDGET: MAX chars is sized in SPEAKING SECONDS, but the real ceiling is the
engine's context window: a chunk costs ~50 audio tokens/sec plus ~1.0 (Hindi)
to ~1.5 (Telugu) prompt tokens/char. At max_model_len=1024 a chunk must stay
under ~13 s / ~180 chars; 25 s / 347 chars needs max_model_len=2048. Set
CHUNK_MAX_CHARS to match the deployed window.
"""
import os
import re

_DASH_ELLIPSIS = dict.fromkeys(map(ord, "â€•â€’â¸ºâ¸»"), " ")   # bar/figure/2-3em dash -> space
                                                          # (em/en dash and â€¦ kept: clause cues)
def post_process(s):
    if not s:
        return s
    s = s.translate(_DASH_ELLIPSIS)
    s = re.sub(r"\.{2,}", "â€¦", s)                # ASCII ".." / "..." -> ellipsis (a sentence cue)
    s = re.sub(r"\s+([,.;:!?à¥¤])", r"\1", s)      # drop space before punctuation
    s = re.sub(r"\s{2,}", " ", s)                # collapse runs of spaces
    return s.strip()


SPC = 0.060                                       # measured speaking rate, s/char: 40 clips,
                                                  # 10 languages x 2 voices, mean 0.058
                                                  # (ta/ml ~0.051, or ~0.062 -- 1.54x spread).
                                                  # 0.072 was a guess and over-split by ~20%.
MAX = int(os.environ.get("CHUNK_MAX_CHARS", str(round(20 / SPC))))   # 333 -> ~20.0 s (fits max_model_len=2048)
MIN = int(os.environ.get("CHUNK_MIN_CHARS", "60"))                   # ~4.3 s: merge runts
HARD = MAX                                        # absolute ceiling (except a lone long token)
CHUNK_GAP_S = 0.15                                # silence inserted between concatenated chunks
SECS_PER_CHAR = SPC                               # back-compat alias
CHUNK_MAX_SEC = MAX * SPC                         # back-compat alias

TAG_G = re.compile(r"\[[a-z][a-z ]*\]")                          # bracketed lowercase = TTS tag
SENT_END = re.compile(r"(?:[.!?à¥¤à¥¥â€¦]|\.{3})+[\"'â€â€™Â»)\]]*$")
CLAUSE_END = re.compile(r"[,;:â€”â€“][\"'â€â€™Â»)\]]*$")
_WS = re.compile(r"\s+")
STRIP = "\"'â€œâ€â€˜â€™Â«Â»()[],.;:!?à¥¤à¥¥â€¦-â€“â€”"
STRIP_LEAD = "\"'â€œâ€â€˜â€™Â«Â»("

CONNECTORS = {
    "and", "but", "because", "while", "so", "then",
    "à¤”à¤°", "à¤à¤µà¤‚", "à¤¤à¤¥à¤¾", "à¤²à¥‡à¤•à¤¿à¤¨", "à¤ªà¤°", "à¤®à¤—à¤°", "à¤•à¥à¤¯à¥‹à¤‚à¤•à¤¿", "à¤œà¤¬à¤•à¤¿", "à¤«à¤¿à¤°", "à¤†à¤£à¤¿", "à¤ªà¤£", "à¤•à¤¾à¤°à¤£",
    "à¦à¦¬à¦‚", "à¦†à¦°", "à¦•à¦¿à¦¨à§à¦¤à§", "à¦•à¦¾à¦°à¦£", "à¦¤à¦¾à¦°à¦ªà¦°", "à¬à¬¬à¬‚", "à¬•à¬¿à¬¨à­à¬¤à­",
    "àª…àª¨à«‡", "àªªàª£", "àª•àª¾àª°àª£", "à¨…à¨¤à©‡", "à¨ªà¨°", "à¨•à¨¿à¨‰à¨‚à¨•à¨¿",
    "à°®à°°à°¿à°¯à±", "à°•à°¾à°¨à±€", "à°Žà°‚à°¦à±à°•à°‚à°Ÿà±‡", "à²®à²¤à³à²¤à³", "à²†à²¦à²°à³†", "à²à²•à³†à²‚à²¦à²°à³†",
    "à®®à®±à¯à®±à¯à®®à¯", "à®†à®©à®¾à®²à¯", "à®à®©à¯†à®©à®¿à®²à¯", "à´ªà´•àµà´·àµ‡", "à´Žà´¨àµà´¨à´¾àµ½", "à´•à´¾à´°à´£à´‚",
}


def _strip(s, chars):
    return s.strip(chars)


def _groups(para):
    """Whitespace tokens -> atomic groups, with emotion tags glued to their word."""
    prot = TAG_G.sub(lambda m: m.group(0).replace(" ", "\x00"), para)
    toks = [t.replace("\x00", " ") for t in prot.split() if t]
    out, glue_right = [], False
    for t in toks:
        bare = _strip(TAG_G.sub("", t).strip(), STRIP)
        is_tag = bare == ""                                   # token is only tag(+punctuation)
        if glue_right:
            out[-1] = out[-1] + " " + t
            glue_right = is_tag                               # consecutive tags keep gluing
            continue
        if is_tag and out and not SENT_END.search(out[-1]):
            out[-1] = out[-1] + " " + t                       # mid/end tag -> preceding word
        elif is_tag:
            out.append(t)                                     # sentence-leading tag -> next word
            glue_right = True
        else:
            out.append(t)
    return out


def _strength(group, nxt):
    """Boundary strength AFTER this group: 3 sentence, 2 clause, 1 word."""
    if SENT_END.search(group):
        return 3
    if CLAUSE_END.search(group):
        return 2
    if nxt is not None:
        parts = nxt.split()
        first = _strip(parts[0], STRIP_LEAD) if parts else ""
        if first in CONNECTORS:
            return 2                                          # break BEFORE a connector word
    return 1


def _runs(gs, min_strength):
    """Join groups into runs, breaking where boundary strength >= min_strength."""
    out, buf = [], []
    for i, g in enumerate(gs):
        buf.append(g)
        nxt = gs[i + 1] if i + 1 < len(gs) else None
        if nxt is not None and _strength(g, nxt) >= min_strength:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out


def _split_words(clause):
    out, buf = [], ""
    for g in _groups(clause):
        cand = (buf + " " + g) if buf else g
        if len(cand) <= HARD or not buf:
            buf = cand
        else:
            out.append(buf)
            buf = g
    if buf:
        out.append(buf)
    return out


def _split_long(sent):
    out, buf = [], ""
    for clause in _runs(_groups(sent), 2):
        if len(clause) > MAX:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_split_words(clause))
            continue
        cand = (buf + " " + clause) if buf else clause
        if len(cand) <= MAX:
            buf = cand
        else:
            out.append(buf)
            buf = clause
    if buf:
        out.append(buf)
    return out


def _chunk_groups(gs):
    out, buf = [], ""
    for sent in _runs(gs, 3):
        if len(sent) > MAX:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_split_long(sent))
            continue
        cand = (buf + " " + sent) if buf else sent
        if len(cand) <= MAX:
            buf = cand
        else:
            out.append(buf)
            buf = sent
    if buf:
        out.append(buf)
    return out


def _merge_runts(pieces):
    if not pieces:
        return []
    merged = []
    for p in pieces:
        if merged and len(p) < MIN and len(merged[-1] + " " + p) <= HARD:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    if len(merged) >= 2 and len(merged[0]) < MIN and len(merged[0] + " " + merged[1]) <= HARD:
        merged[1] = merged[0] + " " + merged[1]
        merged.pop(0)
    return [_WS.sub(" ", m).strip() for m in merged]


def split_for_tts(text, max_sec=None, spc=SPC, hard_sec=None):
    """Split normalized text into prosody-friendly, tag-safe TTS chunks.

    max_sec/spc/hard_sec are accepted for call-site compatibility with the old
    chunker; sizing is governed by MAX/MIN/HARD (chars) above.
    """
    if not text or not text.strip():
        return []
    pieces = []
    for para in text.split("\n"):
        p = para.strip()
        if not p:
            continue
        pieces.extend(_chunk_groups(_groups(p)))
    return _merge_runts(pieces) or [text.strip()]
'''),
    'mithi_tts.tts': (False, r'''"""Mithi text-to-speech: 11 Indian languages, 10 voices: Satvik, Anirban, Nakul, Vasant, Siddhant (male); Charvi, Ahana, Vaidehi, Malini, Jahnavi (female).

    text -> Maya text normaliser -> espeak-ng / misaki G2P (Kokoro phonemes)
         -> indic-bert word vectors -> prosody writer (durations, pitch, loudness, breaths per phoneme)
         -> Kokoro text encoder + iSTFTNet decoder -> 24 kHz audio

    from mithi_tts import Mithi
    tts = Mithi("path/to/mithi-v2-flash")          # the downloaded model folder
    audio = tts.synthesize("à¤†à¤œ à¤®à¥Œà¤¸à¤® à¤•à¤¿à¤¤à¤¨à¤¾ à¤…à¤šà¥à¤›à¤¾ à¤¹à¥ˆ!", lang="hi", voice="charvi")   # numpy array, 24 kHz
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file

from . import frontend as fe
from .indic_encoder import IndicTextEncoder
from .kokoro.istftnet import Decoder
from .kokoro.modules import TextEncoder
from .prosody import P5_COLS, WRITER_LANGS, ProsodyWriter, decode

SR = 24000
from ._here import HERE as DEFAULT_DIR


class KokoroDecoder(nn.Module):
    """The parts of Kokoro-82M this model uses: phoneme text encoder and iSTFTNet decoder."""

    def __init__(self, cfg):
        super().__init__()
        self.text_encoder = TextEncoder(channels=cfg["hidden_dim"], kernel_size=cfg["text_encoder_kernel_size"],
                                        depth=cfg["n_layer"], n_symbols=cfg["n_token"])
        self.decoder = Decoder(dim_in=cfg["hidden_dim"], style_dim=cfg["style_dim"], dim_out=cfg["n_mels"],
                               disable_complex=False, **cfg["istftnet"])


class Mithi:
    LANGS = fe.LANGS

    def __init__(self, model_dir=DEFAULT_DIR, device=None):
        self.dir = model_dir
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.backends.cuda.matmul.allow_tf32 = False
        m = os.path.join(model_dir, "model")
        if not os.path.isdir(m):                               # weights at the top of the model folder
            m = model_dir
        self.cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
        self.settings = dict(self.cfg["inference"])

        kcfg = self.cfg["kokoro"]
        self.vocab = kcfg["vocab"]
        self.kokoro = KokoroDecoder(kcfg)
        self.kokoro.load_state_dict(load_file(os.path.join(m, "decoder.safetensors")))

        with safe_open(os.path.join(m, "writer.safetensors"), "pt") as f:
            meta = f.metadata()
        self.voices = json.loads(meta["voices"])
        self.writer = ProsodyWriter(n_spk=len(self.voices))
        self.writer.load_state_dict(load_file(os.path.join(m, "writer.safetensors")))

        self.text_enc = IndicTextEncoder(os.path.join(m, "indic_encoder"))
        self.kokoro, self.writer, self.text_enc = (x.to(self.dev).eval() for x in (self.kokoro, self.writer, self.text_enc))
        self.packs = {v: t.float().to(self.dev) for v, t in load_file(os.path.join(m, "voices.safetensors")).items()}

    # ------------------------------------------------------------------ one sentence
    @torch.no_grad()
    def synth_sentence(self, text, lang, voice, seed=0):
        s, dev = self.settings, self.dev
        torch.manual_seed(seed)
        norm, ps = fe.phonemes(text, lang)
        ids = torch.LongTensor([0, *[i for i in map(self.vocab.get, ps) if i is not None], 0])
        words = fe.words_of(text, norm)
        tok_word, n_groups = fe.token_words(ids.tolist())
        if n_groups == len(words):
            wv, _ = self.text_enc.word_vectors(" ".join(words))
            tw = torch.tensor(tok_word)
            word_vec, has = wv[tw.clamp(min=0)][None], (tw >= 0)[None].to(dev)
        else:                                               # word / phoneme groups disagree: phonemes only
            word_vec = torch.zeros(1, len(ids), self.text_enc.dim, device=dev)
            has = torch.zeros(1, len(ids), dtype=torch.bool, device=dev)
        x = ids[None].to(dev)
        spk = torch.tensor([self.voices.index(voice)], device=dev)
        H = self.writer.read(x, word_vec, has, spk, torch.tensor([WRITER_LANGS.index(lang)], device=dev),
                             torch.zeros_like(x, dtype=torch.bool))
        head_temp = {k: s["voicing_temp"] for k in ("voiced", "uv0", "uv1")}
        dec_ids, codes = self.writer.sample(H, ids, fe.PAUSE_IDS, temp=s["writer_temp"], top_p=s["writer_top_p"],
                                            head_temp=head_temp)
        codes[[0, -1], 0] = codes[[0, -1], 0].clamp(max=s["edge_max"])      # short silence at both ends
        dur, f0, en = decode(codes[:, :P5_COLS].numpy())
        f0 = f0 * np.float32(2 ** (s["pitch_shift"] / 12))
        y = dec_ids[None].to(dev)
        T = int(dur.sum())
        aln = torch.zeros(1, y.shape[1], T, device=dev)
        aln[0, torch.repeat_interleave(torch.arange(y.shape[1], device=dev), torch.from_numpy(dur).to(dev)),
            torch.arange(T, device=dev)] = 1
        ref = self.packs[voice][min(len(ps), len(self.packs[voice])) - 1]
        h = self.kokoro.text_encoder(y, torch.tensor([y.shape[1]], device=dev), torch.zeros_like(y, dtype=torch.bool)) @ aln
        audio = self.kokoro.decoder(h, torch.from_numpy(f0)[None].to(dev), torch.from_numpy(en)[None].to(dev), ref[:, :128])
        return audio.squeeze().float().cpu().numpy()

    # ------------------------------------------------------------------ any length
    def split(self, text, lang):
        """The pieces synthesize() reads: the whole text, or sentences / comma pieces when it is long."""
        return fe.split_text(text, lang, self.settings["max_phonemes_per_piece"])

    def synthesize(self, text, lang="hi", voice="charvi", seed=None, gap=0.15):
        """Text of any length -> audio at 24 kHz in [-1, 1]. Long text is read sentence by sentence."""
        if lang not in fe.LANGS:
            raise ValueError(f"unknown language {lang} (one of {', '.join(fe.LANGS)})")
        if voice not in self.voices:
            raise ValueError(f"unknown voice {voice} (one of {', '.join(self.voices)})")
        if not text or not text.strip():
            raise ValueError("empty text")
        seed = self.settings["seed"] if seed is None else seed
        silence = np.zeros(int(gap * SR), np.float32)
        segs = []
        for p in self.split(text, lang):
            segs += ([silence] if segs else []) + [self.synth_sentence(p, lang, voice, seed).astype(np.float32)]
        return np.clip(np.concatenate(segs), -1.0, 1.0)

    def save(self, audio, path):
        import soundfile as sf
        sf.write(path, audio, SR, subtype="PCM_16")
'''),
}
_SOURCES["mithi_tts._here"] = (False, "HERE = " + repr(HERE) + "\n")


class _Embedded(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name not in _SOURCES:
            return None
        return importlib.util.spec_from_loader(name, self, is_package=_SOURCES[name][0])

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        is_pkg, src = _SOURCES[module.__name__]
        fname = os.path.join(HERE, *module.__name__.split(".")) + ("/__init__.py" if is_pkg else ".py")
        module.__file__ = fname
        if is_pkg:
            module.__path__ = [os.path.dirname(fname)]
        linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)     # readable tracebacks
        exec(compile(src, fname, "exec"), module.__dict__)


if not any(isinstance(f, _Embedded) for f in sys.meta_path):
    sys.meta_path.insert(0, _Embedded())

from mithi_tts import SR, Mithi  # noqa: E402,F401


# ---------------------------------------------------------------- command line
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True, help="text to speak")
    ap.add_argument("--lang", default="hi", help="hi en bn gu kn ml mr or pa ta te")
    ap.add_argument("--voice", default="charvi", help="satvik anirban nakul vasant siddhant (male), charvi ahana vaidehi malini jahnavi (female)")
    ap.add_argument("--out", default="output.wav", help="output .wav file")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: PyTorch's choice, usually all cores)")
    args = ap.parse_args()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    tts = Mithi(HERE, device=args.device)
    t0 = time.time()
    audio = tts.synthesize(args.text, lang=args.lang, voice=args.voice)
    tts.save(audio, args.out)
    print(f"{args.out}: {len(audio) / SR:.2f} s of audio in {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
