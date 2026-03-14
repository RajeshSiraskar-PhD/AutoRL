from __future__ import annotations

import argparse
from pathlib import Path

import fitz


DEFAULT_FOLDERS = (
    Path("/Users/rajeshsiraskar/PhD/Code/AutoRL/paper/figures and reports/IEEE_Results"),
    Path("/Users/rajeshsiraskar/PhD/Code/AutoRL/paper/figures and reports/SIT_Results"),
)


def convert_pdf_to_jpg(pdf_path: Path, dpi: int) -> Path:
    jpg_path = pdf_path.with_suffix(".jpg")
    zoom = dpi / 72.0

    with fitz.open(pdf_path) as document:
        if document.page_count == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        page = document.load_page(0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pixmap.pil_save(jpg_path, format="JPEG", dpi=(dpi, dpi))

    return jpg_path


def convert_folder(folder: Path, dpi: int) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {folder}")

    created_files: list[Path] = []
    for pdf_path in sorted(folder.glob("*.pdf")):
        created_files.append(convert_pdf_to_jpg(pdf_path, dpi=dpi))
    return created_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all PDF files in the target folders to 300 DPI JPG files.",
    )
    parser.add_argument(
        "folders",
        nargs="*",
        type=Path,
        default=DEFAULT_FOLDERS,
        help="Folders containing PDF files. Defaults to the IEEE and SIT result folders.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution in dots per inch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total_created = 0
    for folder in args.folders:
        created_files = convert_folder(folder, dpi=args.dpi)
        for output_path in created_files:
            print(f"Created {output_path}")
        total_created += len(created_files)

    print(f"Converted {total_created} PDF file(s) to JPG at {args.dpi} DPI.")


if __name__ == "__main__":
    main()