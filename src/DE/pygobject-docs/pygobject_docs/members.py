from itertools import chain

from gi._gi import SignalInfo, VFuncInfo
from gi._signature import get_pytype


def own_dir(obj_type: type) -> list[str]:
    # Find all elements of a type, that are part of the type
    # and not of a parent or interface.

    # GObject.Object is our base type - show all members
    if (
        obj_type.__module__ in ("gi.overrides.GObject", "gi.repository.GObject")
        and obj_type.__name__ == "Object"
    ):
        return dir(obj_type)

    if getattr(obj_type, "__overridden__", None) is obj_type or obj_type.__module__.startswith(
        "gi.overrides"
    ):
        return sorted(set(chain(obj_type.__dict__.keys(), obj_type.__base__.__dict__.keys())))

    return sorted(obj_type.__dict__.keys())


def properties(obj_type: type) -> list[tuple[str, object | type]]:
    try:
        props = obj_type.__info__.get_properties()  # type: ignore[attr-defined]
    except AttributeError:
        return []

    return sorted((p.get_name(), get_pytype(p.get_type_info())) for p in props)


def virtual_methods(obj_type: type) -> list[VFuncInfo]:
    try:
        vfuncs = obj_type.__info__.get_vfuncs()  # type: ignore[attr-defined]
    except AttributeError:
        return []

    return sorted(vfuncs, key=lambda v: v.get_name())


def signals(obj_type: type) -> list[SignalInfo]:
    try:
        sigs = obj_type.__info__.get_signals()  # type: ignore[attr-defined]
    except AttributeError:
        return []

    return sorted(sigs, key=lambda s: s.get_name())
