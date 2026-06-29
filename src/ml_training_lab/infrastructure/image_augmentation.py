from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import pandas as pd


DEFAULT_AUGMENTATION_PARAMS = {
    "horizontal_flip_p": 0.5,
    "vertical_flip_p": 0.5,
    "rotate_limit": 12,
    "rotate_p": 0.6,
    "brightness_contrast_p": 0.35,
    "gauss_noise_p": 0.25,
    "blur_p": 0.15,
}


def build_augmentation_pipeline(params: dict, seed: int) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=params["horizontal_flip_p"]),
            A.VerticalFlip(p=params["vertical_flip_p"]),
            A.Rotate(limit=params["rotate_limit"], border_mode=cv2.BORDER_REFLECT_101, p=params["rotate_p"]),
            A.RandomBrightnessContrast(p=params["brightness_contrast_p"]),
            A.GaussNoise(p=params["gauss_noise_p"]),
            A.Blur(blur_limit=3, p=params["blur_p"]),
        ],
        seed=seed,
    )


def enrich_with_augmentations(
    df: pd.DataFrame,
    *,
    image_root: Path,
    sludge_image_column: str,
    lba_image_column: str,
    is_augmented_column: str,
    sludge_augmented_folder: str,
    lba_augmented_folder: str,
    augmentations_per_record: int,
    random_seed: int,
) -> pd.DataFrame:
    if augmentations_per_record <= 0:
        raise ValueError("augmentations_per_record must be greater than zero.")
    _validate_metadata(df, is_augmented_column)
    clean_df = df.loc[df[is_augmented_column].eq(False)].copy().reset_index(drop=True)
    sludge_pipeline = build_augmentation_pipeline(DEFAULT_AUGMENTATION_PARAMS, random_seed)
    lba_pipeline = build_augmentation_pipeline(DEFAULT_AUGMENTATION_PARAMS, random_seed + 1)
    augmented_rows = []
    for row_index, row in clean_df.iterrows():
        sludge_image = _load_rgb_image(image_root / row[sludge_image_column])
        lba_image = _load_rgb_image(image_root / row[lba_image_column])
        for augmentation_index in range(1, augmentations_per_record + 1):
            sludge_augmented = sludge_pipeline(image=sludge_image)["image"]
            lba_augmented = lba_pipeline(image=lba_image)["image"]
            augmented_row = row.copy()
            augmented_row[sludge_image_column] = _augmented_relative_path(
                row[sludge_image_column], sludge_augmented_folder, row_index, augmentation_index
            )
            augmented_row[lba_image_column] = _augmented_relative_path(
                row[lba_image_column], lba_augmented_folder, row_index, augmentation_index
            )
            augmented_row[is_augmented_column] = True
            _save_rgb_image(image_root / augmented_row[sludge_image_column], sludge_augmented)
            _save_rgb_image(image_root / augmented_row[lba_image_column], lba_augmented)
            augmented_rows.append(augmented_row)
    if not augmented_rows:
        return clean_df
    return pd.concat([clean_df, pd.DataFrame(augmented_rows)], ignore_index=True)


def _validate_metadata(df: pd.DataFrame, is_augmented_column: str) -> None:
    if is_augmented_column not in df.columns:
        raise ValueError(f"Column '{is_augmented_column}' is required.")
    if df[is_augmented_column].isna().any():
        raise ValueError(f"Column '{is_augmented_column}' contains missing values.")


def _load_rgb_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image does not exist or cannot be read: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _save_rgb_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _augmented_relative_path(source_relative_path: str, target_folder: str, row_index: int, augmentation_index: int) -> str:
    source_path = Path(source_relative_path)
    return str(
        Path(target_folder)
        / f"{source_path.stem}__row_{row_index:06d}__alb_{augmentation_index:02d}{source_path.suffix}"
    )
