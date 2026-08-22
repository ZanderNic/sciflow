# std lib imports
from pathlib import Path
import gzip
import shutil

# 3-party imports
from remotezip import RemoteZip



def download_scifate_files(
    output_dir="../data/ScifateData",
):
    url = "https://zenodo.org/records/14176698/files/SciFate.zip?download=1"
    base = "SciFate/grand3/scifate.targets/"

    wanted = {
        "4sU.Binom.ntr.mtx.gz",
        "barcodes.tsv.gz",
        "features.tsv.gz",
        "matrix.mtx.gz",
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with RemoteZip(url) as archive:
        for filename in wanted:
            output_path = output_dir / filename.removesuffix(".gz")

            if output_path.exists():
                continue

            print(f"Downloading {filename}...")

            with archive.open(base + filename) as src:
                with gzip.open(src, "rb") as compressed:
                    with output_path.open("wb") as dst:
                        shutil.copyfileobj(compressed, dst)

    print("SciFate files ready.")