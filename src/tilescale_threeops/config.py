from dataclasses import dataclass


@dataclass(frozen=True)
class PathConfig:
    """Enable each retained path independently."""

    ag: bool = False
    rs: bool = False
    ar: bool = False

    @classmethod
    def from_name(cls, name):
        if name == "original":
            return cls()
        parts = name.split("_")
        if parts[0] != "paths" or parts[1:] not in [
            ["ag"],
            ["rs"],
            ["ar"],
            ["ag", "rs"],
            ["ag", "ar"],
            ["rs", "ar"],
            ["ag", "rs", "ar"],
        ]:
            raise ValueError(f"Unknown path configuration {name!r}")
        return cls(ag="ag" in parts, rs="rs" in parts, ar="ar" in parts)

    @property
    def name(self):
        parts = [name for name in ("ag", "rs", "ar") if getattr(self, name)]
        return "paths_" + "_".join(parts) if parts else "original"
