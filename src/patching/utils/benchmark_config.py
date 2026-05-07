
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_CONFIG_DIR = REPO_ROOT / "benchmark_configs"


@dataclass(frozen=True)
class BenchmarkMethodConfig:
    """Human-readable benchmark config for one steering method."""

    method_name: str
    path: Path
    training_fixed: dict[str, Any] = field(default_factory=dict)
    validation_fixed: dict[str, Any] = field(default_factory=dict)
    validation_selected: dict[str, Any] = field(default_factory=dict)
    inference_fixed: dict[str, Any] = field(default_factory=dict)

    def fixed_params(self, phase_name: str) -> dict[str, Any]:
        if phase_name == "training":
            return dict(self.training_fixed)
        if phase_name == "validation":
            return dict(self.validation_fixed)
        if phase_name == "inference":
            return dict(self.inference_fixed)
        raise ValueError(
            f"Unsupported phase {phase_name!r}. Expected 'training', 'validation', or 'inference'."
        )

    def selected_params(self) -> dict[str, Any]:
        return _selection_defaults(self.validation_selected)


def default_benchmark_config_path(method_name: str) -> Path:
    """Return the default YAML config path for a steering method."""

    return BENCHMARK_CONFIG_DIR / f"{method_name}.yaml"


def load_benchmark_method_config(
    method_name: str,
    config_path: str | Path | None = None,
) -> BenchmarkMethodConfig:
    """Load one benchmark config YAML."""

    path = default_benchmark_config_path(method_name) if config_path is None else Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {path}")

    payload = _load_config_payload(path)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError(f"Benchmark config must be a YAML mapping: {path}")

    config_method_name = str(payload.get("method", method_name))
    if config_method_name != method_name:
        raise ValueError(
            f"Benchmark config {path} declares method={config_method_name!r}, "
            f"but {method_name!r} was requested."
        )

    return BenchmarkMethodConfig(
        method_name=config_method_name,
        path=path,
        training_fixed=_dict_section(payload.get("training_fixed"), section_name="training_fixed"),
        validation_fixed=_dict_section(payload.get("validation_fixed"), section_name="validation_fixed"),
        validation_selected=_dict_section(payload.get("validation_selected"), section_name="validation_selected"),
        inference_fixed=_dict_section(payload.get("inference_fixed"), section_name="inference_fixed"),
    )


def load_benchmark_method_config_from_path(config_path: str | Path) -> BenchmarkMethodConfig:
    """Load one benchmark config and infer the method from its contents."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {path}")

    payload = _load_config_payload(path)
    if payload is None or not isinstance(payload, dict):
        raise TypeError(f"Benchmark config must be a YAML mapping: {path}")

    method_name = payload.get("method")
    if not isinstance(method_name, str) or not method_name:
        raise ValueError(f"Benchmark config must define a non-empty 'method' field: {path}")
    return load_benchmark_method_config(method_name=method_name, config_path=path)


def _dict_section(payload: Any, *, section_name: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Section {section_name!r} must be a mapping.")
    return dict(payload)


def _selection_defaults(specs: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for name, spec in specs.items():
        if isinstance(spec, dict):
            if "default" in spec:
                defaults[name] = spec["default"]
            continue
        if spec is not None:
            defaults[name] = spec
    return defaults


def _load_config_payload(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)
