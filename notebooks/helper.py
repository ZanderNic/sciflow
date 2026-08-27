# std lib imports
from pathlib import Path
import gzip
import shutil
import csv

# 3-party imports
from remotezip import RemoteZip
import pandas as pd
import numpy as np
import tempfile
import pyreadr







from pathlib import Path
import csv
import gzip
import shutil
import tempfile

import pyreadr
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

    trajectory_file = "SciFate/simulation/trajectories_groundTruth.rds"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with RemoteZip(url) as archive:

        # Download expression / NTR files
        for filename in wanted:
            output_path = output_dir / filename.removesuffix(".gz")

            if output_path.exists():
                continue

            print(f"Downloading {filename}...")

            with archive.open(base + filename) as src:
                with gzip.open(src, "rb") as compressed:
                    with output_path.open("wb") as dst:
                        shutil.copyfileobj(compressed, dst)

        # Download and convert trajectory file
        trajectory_output = output_dir / "trajectories_sci.csv"

        if not trajectory_output.exists():
            print("Downloading trajectories_groundTruth.rds...")

            with archive.open(trajectory_file) as src:
                with tempfile.NamedTemporaryFile(suffix=".rds") as tmp:
                    shutil.copyfileobj(src, tmp)
                    tmp.flush()

                    result = pyreadr.read_r(tmp.name)

            if not result:
                raise ValueError(
                    "No readable object found in trajectory RDS file."
                )

            trajectories = next(iter(result.values()))

            trajectories.index = range(1, len(trajectories) + 1)

            trajectories.to_csv(trajectory_output, index=True, index_label="", quoting=csv.QUOTE_ALL)

            print(f"Saved trajectories to {trajectory_output}")

    print("SciFate files ready.")
    
    
def assignments_to_df(assignments):
    labels = (
        assignments["cluster"].to_numpy()
        if isinstance(assignments, pd.DataFrame)
        else np.asarray(assignments).ravel()
    )

    return pd.DataFrame({
        "id": np.arange(len(labels)),
        "cluster": labels,
    })
    