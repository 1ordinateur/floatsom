"""Convert config dataclasses to and from plain JSON-safe dictionaries."""

import dataclasses
import difflib
import typing
from typing import Any, Dict, Mapping

import numpy as np


def dataclass_to_dict(instance, exclude=frozenset()) -> Dict[str, Any]:
    """Return the fields of ``instance`` as a nested dict of plain values."""
    return {
        field.name: _to_plain(getattr(instance, field.name))
        for field in dataclasses.fields(instance)
        if field.name not in exclude
    }


def dataclass_from_dict(
    cls,
    data: Mapping[str, Any],
    *,
    exclude=frozenset(),
    path: str = "",
):
    """Build ``cls`` from ``data``, and build nested dataclasses from nested dicts."""
    if not isinstance(data, Mapping):
        raise ValueError(
            f"{path or cls.__name__} must be a mapping of setting names to values, "
            f"got {type(data).__name__}."
        )
    fields = {
        field.name: field
        for field in dataclasses.fields(cls)
        if field.init and field.name not in exclude
    }
    _reject_unknown_keys(data, fields, path)

    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        hint = hints.get(name)
        nested_cls = _dataclass_in_hint(hint)
        key_path = f"{path}.{name}" if path else name
        if nested_cls is not None and value is None and nested_cls is hint:
            raise ValueError(
                f"{key_path} must be a mapping of setting names to values, not None. "
                "Leave it out to use the defaults."
            )
        if nested_cls is not None and value is not None and not isinstance(value, nested_cls):
            value = dataclass_from_dict(nested_cls, value, path=key_path)
        kwargs[name] = value
    try:
        return cls(**kwargs)
    except TypeError as error:
        raise ValueError(f"{path or cls.__name__}: {error}") from error


def _reject_unknown_keys(data, fields, path):
    unknown = [key for key in data if key not in fields]
    if not unknown:
        return
    lines = []
    for key in unknown:
        key_path = f"{path}.{key}" if path else str(key)
        matches = difflib.get_close_matches(str(key), list(fields), n=1)
        hint = f" Did you mean '{matches[0]}'?" if matches else ""
        lines.append(f"Unknown setting '{key_path}'.{hint}")
    lines.append(f"Valid settings{' in ' + path if path else ''}: {', '.join(sorted(fields))}.")
    raise ValueError(" ".join(lines))


def _dataclass_in_hint(hint):
    """Return the dataclass in ``hint`` (``X`` or ``Optional[X]``), or None."""
    if hint is None:
        return None
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        return hint
    if typing.get_origin(hint) is typing.Union:
        for argument in typing.get_args(hint):
            if dataclasses.is_dataclass(argument) and isinstance(argument, type):
                return argument
    return None


def _to_plain(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclass_to_dict(value)
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
