from __future__ import annotations

import random
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
SAMPLED_DIR = ROOT / "sampled_datasets"
CLASSIFICATION_CSV = ROOT / "game_midi_loop_classification.csv"

SAMPLE_SIZE = 250
RANDOM_SEED = 42

GAME_LABEL_TO_GENRE = {
    "looping_gameplay_bgm": "game_looping",
    "finite_song_or_cue": "game_not_looping",
}

GENRE_SPECS = {
    "classical": {
        "source_dir": DATASETS_DIR / "classical_midi",
        "target_dir": SAMPLED_DIR / "classical_midi",
    },
    "pop": {
        "source_dir": DATASETS_DIR / "pop_midi",
        "target_dir": SAMPLED_DIR / "pop_midi",
    },
    "game_looping": {
        "target_dir": SAMPLED_DIR / "game_looping_midi",
    },
    "game_not_looping": {
        "target_dir": SAMPLED_DIR / "game_not_looping_midi",
    },
}


def midi_files_in(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
    )


def reset_folder(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    for path in folder.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()


def copy_sample(files: list[Path], target_dir: Path, rng: random.Random) -> None:
    if len(files) < SAMPLE_SIZE:
        raise ValueError(
            f"Need at least {SAMPLE_SIZE} files for {target_dir.name}, found {len(files)}."
        )

    reset_folder(target_dir)

    for source in rng.sample(files, SAMPLE_SIZE):
        shutil.copy2(source, target_dir / source.name)


def sample_classical_and_pop(rng: random.Random) -> None:
    for genre in ("classical", "pop"):
        spec = GENRE_SPECS[genre]
        source_files = midi_files_in(spec["source_dir"])
        copy_sample(source_files, spec["target_dir"], rng)


def sample_game_subgenres(rng: random.Random) -> None:
    classification = pd.read_csv(CLASSIFICATION_CSV)

    for label, genre in GAME_LABEL_TO_GENRE.items():
        spec = GENRE_SPECS[genre]

        relative_paths = classification.loc[
            classification["label"] == label, "file"
        ].tolist()

        source_files = [
            DATASETS_DIR / Path(relative_path)
            for relative_path in relative_paths
        ]

        missing_files = [path for path in source_files if not path.exists()]
        if missing_files:
            preview = ", ".join(path.name for path in missing_files[:5])
            raise FileNotFoundError(
                f"Missing {len(missing_files)} files for {genre}: {preview}"
            )

        copy_sample(source_files, spec["target_dir"], rng)


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    sample_classical_and_pop(rng)
    sample_game_subgenres(rng)

    print("Resampled 250 files for each genre:")
    for genre, spec in GENRE_SPECS.items():
        count = len(midi_files_in(spec["target_dir"]))
        print(f"  {genre}: {count}")


if __name__ == "__main__":
    main()
