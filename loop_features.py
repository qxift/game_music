from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pretty_midi


PITCH_CLASS_NAMES = np.array(["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"])


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _repetition_ratio(items: Iterable[tuple]) -> float:
    items = list(items)
    if not items:
        return float("nan")
    counts = Counter(items)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(items)


def _collect_melodic_notes(pm: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    notes: list[pretty_midi.Note] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    return notes


def _get_beat_grid(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    end_time = max(pm.get_end_time(), 1e-6)
    beats = np.asarray(pm.get_beats(), dtype=float)

    if beats.size < 2:
        tempi, _ = pm.get_tempo_changes()
        tempo = float(np.median(tempi)) if len(tempi) else 120.0
        beat_duration = 60.0 / max(tempo, 1e-6)
        beats = np.arange(0.0, end_time + beat_duration, beat_duration)

    beats = np.unique(np.clip(beats, 0.0, end_time))
    if beats.size == 0 or beats[0] > 0.0:
        beats = np.insert(beats, 0, 0.0)
    if beats[-1] < end_time:
        beats = np.append(beats, end_time)
    return beats


def _get_window_edges(pm: pretty_midi.PrettyMIDI, beats_per_window: int = 4) -> np.ndarray:
    beats = _get_beat_grid(pm)
    end_time = max(pm.get_end_time(), 1e-6)

    if beats.size <= beats_per_window:
        return np.array([0.0, end_time], dtype=float)

    edges = beats[::beats_per_window]
    if edges[0] != 0.0:
        edges = np.insert(edges, 0, 0.0)
    if edges[-1] < end_time:
        edges = np.append(edges, end_time)
    return np.unique(edges)


def _window_overlap(note: pretty_midi.Note, start: float, end: float) -> float:
    return max(0.0, min(note.end, end) - max(note.start, start))


def _window_signature(notes: list[pretty_midi.Note], start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    chroma = np.zeros(12, dtype=float)
    onset_chroma = np.zeros(12, dtype=float)
    note_count = 0.0
    onset_count = 0.0
    pitch_sum = 0.0

    for note in notes:
        overlap = _window_overlap(note, start, end)
        if overlap <= 0.0:
            continue
        pc = note.pitch % 12
        chroma[pc] += overlap
        note_count += 1.0
        pitch_sum += note.pitch
        if start <= note.start < end:
            onset_chroma[pc] += 1.0
            onset_count += 1.0

    window_duration = max(end - start, 1e-6)
    chroma = chroma / max(chroma.sum(), 1e-6)
    onset_chroma = onset_chroma / max(onset_chroma.sum(), 1e-6)
    pitch_center = pitch_sum / max(note_count, 1.0)
    extra = np.array(
        [
            note_count / window_duration,
            onset_count / window_duration,
            pitch_center / 127.0,
        ],
        dtype=float,
    )
    return np.concatenate([chroma, onset_chroma, extra]), chroma


def _match_template(chroma: np.ndarray) -> tuple[int | None, str | None, float]:
    if chroma.sum() <= 0.0:
        return None, None, 0.0

    templates = []
    for root in range(12):
        major = np.zeros(12, dtype=float)
        major[[root, (root + 4) % 12, (root + 7) % 12]] = 1.0
        minor = np.zeros(12, dtype=float)
        minor[[root, (root + 3) % 12, (root + 7) % 12]] = 1.0
        templates.append((root, "maj", major))
        templates.append((root, "min", minor))

    best_root = None
    best_quality = None
    best_score = -1.0
    for root, quality, template in templates:
        score = _cosine_similarity(chroma, template)
        if score > best_score:
            best_root = root
            best_quality = quality
            best_score = score

    if best_score < 0.45:
        return None, None, float(best_score)
    return best_root, best_quality, float(best_score)


def _analyze_repeated_sections(
    signatures: list[np.ndarray],
    edges: np.ndarray,
    total_duration: float,
    min_windows: int = 2,
    similarity_threshold: float = 0.92,
) -> dict[str, float]:
    n_windows = len(signatures)
    if n_windows < min_windows * 2:
        return {
            "repeated_section_count": 0,
            "repeated_window_ratio": 0.0,
            "longest_repeated_span_sec": 0.0,
            "longest_repeated_span_ratio": 0.0,
            "repeat_gap_sec": float("nan"),
            "repeat_gap_ratio": float("nan"),
            "mean_repeat_similarity": float("nan"),
        }

    matches = []
    max_windows = n_windows // 2
    for block_len in range(min_windows, max_windows + 1):
        for left in range(0, n_windows - (2 * block_len) + 1):
            for right in range(left + block_len, n_windows - block_len + 1):
                sims = [
                    _cosine_similarity(signatures[left + offset], signatures[right + offset])
                    for offset in range(block_len)
                ]
                block_similarity = float(np.mean(sims))
                if block_similarity < similarity_threshold:
                    continue
                left_start = float(edges[left])
                left_end = float(edges[left + block_len])
                right_start = float(edges[right])
                right_end = float(edges[right + block_len])
                matches.append(
                    {
                        "left": left,
                        "right": right,
                        "block_len": block_len,
                        "similarity": block_similarity,
                        "length_sec": left_end - left_start,
                        "gap_sec": right_start - left_end,
                        "covered": set(range(left, left + block_len)) | set(range(right, right + block_len)),
                    }
                )

    if not matches:
        return {
            "repeated_section_count": 0,
            "repeated_window_ratio": 0.0,
            "longest_repeated_span_sec": 0.0,
            "longest_repeated_span_ratio": 0.0,
            "repeat_gap_sec": float("nan"),
            "repeat_gap_ratio": float("nan"),
            "mean_repeat_similarity": float("nan"),
        }

    best = max(matches, key=lambda item: (item["length_sec"], item["similarity"]))
    covered_windows = set()
    for match in matches:
        covered_windows.update(match["covered"])

    return {
        "repeated_section_count": int(len(matches)),
        "repeated_window_ratio": float(len(covered_windows) / n_windows),
        "longest_repeated_span_sec": float(best["length_sec"]),
        "longest_repeated_span_ratio": float(best["length_sec"] / total_duration),
        "repeat_gap_sec": float(best["gap_sec"]),
        "repeat_gap_ratio": float(best["gap_sec"] / total_duration),
        "mean_repeat_similarity": float(np.mean([match["similarity"] for match in matches])),
    }


def extract_loop_features(path_or_pm: str | pretty_midi.PrettyMIDI, beats_per_window: int = 4) -> dict[str, float | str]:
    pm = path_or_pm if isinstance(path_or_pm, pretty_midi.PrettyMIDI) else pretty_midi.PrettyMIDI(str(path_or_pm))
    notes = _collect_melodic_notes(pm)
    total_duration = max(pm.get_end_time(), 1e-6)

    if not notes:
        return {
            "loop_window_count": 0,
            "repeated_section_count": 0,
            "repeated_window_ratio": 0.0,
            "longest_repeated_span_sec": 0.0,
            "longest_repeated_span_ratio": 0.0,
            "repeat_gap_sec": float("nan"),
            "repeat_gap_ratio": float("nan"),
            "mean_repeat_similarity": float("nan"),
            "chord_loop_closure_similarity": float("nan"),
            "two_step_loop_similarity": float("nan"),
            "closing_matches_opening_chord": 0.0,
            "closing_contains_opening_root": 0.0,
            "penultimate_to_opening_fifth_resolution": 0.0,
            "opening_chord": "NA",
            "closing_chord": "NA",
        }

    edges = _get_window_edges(pm, beats_per_window=beats_per_window)
    signatures = []
    chroma_windows = []
    for start, end in zip(edges[:-1], edges[1:]):
        signature, chroma = _window_signature(notes, float(start), float(end))
        signatures.append(signature)
        chroma_windows.append(chroma)

    repeat_features = _analyze_repeated_sections(signatures, edges, total_duration)

    first_chroma = chroma_windows[0]
    last_chroma = chroma_windows[-1]
    penultimate_chroma = chroma_windows[-2] if len(chroma_windows) >= 2 else np.zeros(12, dtype=float)
    chord_matches = [_match_template(chroma) for chroma in chroma_windows]

    opening_root, opening_quality, _ = _match_template(first_chroma)
    closing_root, closing_quality, _ = _match_template(last_chroma)
    penultimate_root, _, _ = _match_template(penultimate_chroma)

    opening_label = (
        f"{PITCH_CLASS_NAMES[opening_root]}:{opening_quality}" if opening_root is not None and opening_quality else "unknown"
    )
    closing_label = (
        f"{PITCH_CLASS_NAMES[closing_root]}:{closing_quality}" if closing_root is not None and closing_quality else "unknown"
    )

    closing_contains_opening_root = 0.0
    if opening_root is not None and last_chroma.sum() > 0.0:
        closing_contains_opening_root = float(last_chroma[opening_root] >= np.percentile(last_chroma[last_chroma > 0], 50)) if np.any(last_chroma > 0) else 0.0

    penultimate_to_opening = 0.0
    if opening_root is not None and penultimate_root is not None:
        penultimate_to_opening = float((penultimate_root - opening_root) % 12 == 7)

    two_step_similarity = float("nan")
    if len(chroma_windows) >= 2:
        two_step_similarity = float(
            np.mean(
                [
                    _cosine_similarity(penultimate_chroma, first_chroma),
                    _cosine_similarity(last_chroma, first_chroma),
                ]
            )
        )

    return {
        "loop_window_count": int(len(signatures)),
        **repeat_features,
        "chord_loop_closure_similarity": float(_cosine_similarity(first_chroma, last_chroma)),
        "two_step_loop_similarity": float(two_step_similarity),
        "closing_matches_opening_chord": float(
            opening_root is not None
            and closing_root is not None
            and opening_root == closing_root
            and opening_quality == closing_quality
        ),
        "closing_contains_opening_root": closing_contains_opening_root,
        "penultimate_to_opening_fifth_resolution": penultimate_to_opening,
        "opening_chord": opening_label,
        "closing_chord": closing_label,
        "window_chord_repetition_ratio": float(
            _repetition_ratio(tuple((root, quality) for root, quality, _ in chord_matches if root is not None))
        ),
    }
