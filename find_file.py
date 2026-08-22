from remotezip import RemoteZip

url = "https://zenodo.org/records/14176698/files/SciFate.zip?download=1"

# One barcode that is known to occur in your trajectory file
target_barcode = "A549.0h.CCGGGCGCTCGACG"

text_extensions = (
    ".csv",
    ".tsv",
    ".txt",
    ".r",
    ".rmd",
)

with RemoteZip(url) as archive:
    names = archive.namelist()

    print(f"Archive contains {len(names):,} files.\n")

    # ---------------------------------------------------------
    # 1. Look for suspicious filenames
    # ---------------------------------------------------------
    print("=" * 80)
    print("FILES WITH TRAJECTORY-RELATED NAMES")
    print("=" * 80)

    keywords = [
        "traj",
        "trajectory",
        "path",
        "flow",
        "match",
        "prev",
        "link",
    ]

    filename_matches = []

    for name in names:
        lower = name.lower()

        if any(keyword in lower for keyword in keywords):
            filename_matches.append(name)
            print(name)

    if not filename_matches:
        print("No obvious trajectory-related filenames found.")

    # ---------------------------------------------------------
    # 2. List candidate text/data files
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("CSV / TSV / TXT FILES")
    print("=" * 80)

    data_candidates = [
        name
        for name in names
        if name.lower().endswith((".csv", ".tsv", ".txt"))
    ]

    for name in data_candidates:
        print(name)

    print(f"\nFound {len(data_candidates)} CSV/TSV/TXT files.")

    # ---------------------------------------------------------
    # 3. Search these files for a known trajectory barcode
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"SEARCHING FOR BARCODE: {target_barcode}")
    print("=" * 80)

    barcode_matches = []

    for i, name in enumerate(data_candidates, start=1):
        print(
            f"[{i}/{len(data_candidates)}] Checking {name}",
            end="\r"
        )

        try:
            with archive.open(name) as file:
                # Read incrementally so we do not unnecessarily
                # load huge files completely into memory.
                for raw_line in file:
                    try:
                        line = raw_line.decode(
                            "utf-8",
                            errors="ignore"
                        )
                    except AttributeError:
                        line = str(raw_line)

                    if target_barcode in line:
                        barcode_matches.append(name)

                        print(
                            f"\nFOUND BARCODE IN:\n{name}\n"
                        )
                        break

        except Exception as error:
            print(
                f"\nCould not inspect {name}: "
                f"{type(error).__name__}: {error}"
            )

    print("\n" + "=" * 80)
    print("BARCODE SEARCH RESULTS")
    print("=" * 80)

    if barcode_matches:
        for name in barcode_matches:
            print(name)
    else:
        print(
            "The barcode was not found in any CSV/TSV/TXT file."
        )

    # ---------------------------------------------------------
    # 4. If not found, search R/Rmd scripts for trajectory code
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("SEARCHING R / RMD FILES FOR TRAJECTORY CODE")
    print("=" * 80)

    script_candidates = [
        name
        for name in names
        if name.lower().endswith((".r", ".rmd"))
    ]

    script_keywords = [
        "trajectory",
        "trajectories",
        "mincost",
        "min.cost",
        "maxflow",
        "max.flow",
        "prevRNA",
        "prev.rna",
        "matching",
    ]

    script_matches = {}

    for i, name in enumerate(script_candidates, start=1):
        print(
            f"[{i}/{len(script_candidates)}] Checking {name}",
            end="\r"
        )

        try:
            with archive.open(name) as file:
                content = file.read().decode(
                    "utf-8",
                    errors="ignore"
                )

            found_keywords = [
                keyword
                for keyword in script_keywords
                if keyword.lower() in content.lower()
            ]

            if found_keywords:
                script_matches[name] = found_keywords

        except Exception as error:
            print(
                f"\nCould not inspect {name}: "
                f"{type(error).__name__}: {error}"
            )

    print("\n" + "=" * 80)
    print("R / RMD FILES CONTAINING RELEVANT TERMS")
    print("=" * 80)

    if script_matches:
        for name, found_keywords in script_matches.items():
            print(f"\n{name}")
            print(
                "  keywords:",
                ", ".join(found_keywords)
            )
    else:
        print("No relevant R/Rmd scripts found.")