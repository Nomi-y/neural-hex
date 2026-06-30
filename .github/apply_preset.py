#!/usr/bin/env python3
"""Apply a hardware preset from hyperparams.toml, in place.

Usage:  apply_preset.py <preset-name>

Merges the named [presets.<name>] section into the base config sections and
rewrites hyperparams.toml with the merged values baked in (the [presets.*]
tables are stripped). The CI image build runs this so the container ships with
the chosen configuration. Mirrors the preset logic in build_container.sh.
"""
import sys
import tomllib

SECTIONS = ("game", "net", "mcts", "selfplay", "train", "engine")


def fmt(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def main(name: str) -> None:
    with open("hyperparams.toml", "rb") as f:
        data = tomllib.load(f)

    preset = data.get("presets", {}).get(name)
    if preset is None:
        available = list(data.get("presets", {}))
        sys.exit(f'ERROR: preset "{name}" not found. Available: {available}')

    print(f'Applying preset "{name}": {preset.get("description", "(no description)")}')
    for section in SECTIONS:
        if section in preset:
            data.setdefault(section, {})
            for key, value in preset[section].items():
                old = data[section].get(key, "(unset)")
                data[section][key] = value
                print(f"  {section}.{key}: {old} -> {value}")

    with open("hyperparams.toml", "w") as f:
        for section in SECTIONS:
            if section not in data:
                continue
            f.write(f"[{section}]\n")
            for key, value in data[section].items():
                f.write(f"{key} = {fmt(value)}\n")
            f.write("\n")

    print("hyperparams.toml updated.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: apply_preset.py <preset-name>")
    main(sys.argv[1])
