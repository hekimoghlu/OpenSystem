"""Our own inspect, it can inspect normal Python
methods, as well as gi objects.
"""

import logging
from typing import Callable
from inspect import Signature
from re import match

from gi.repository import GLib, GObject
from sphinx.util.docstrings import prepare_docstring
from sphinx.util.inspect import getdoc, signature as sphinx_signature, stringify_signature

from pygobject_docs import overrides


log = logging.getLogger(__name__)

Signature.__str__ = lambda self: stringify_signature(self, unqualified_typehints=True)  # type: ignore[method-assign]


def patch_gi_overrides():
    import gi.overrides
    from gi.overrides import override as real_override

    def override(type_):
        namespace = type_.__module__.rsplit(".", 1)[-1]
        new_type = real_override(type_)
        log.info("%s is overridden", new_type)
        new_type.__module__ = "gi.repository." + namespace
        new_type.__overridden__ = new_type
        return new_type

    gi.overrides.override = override

    # Fix already loaded types

    def fix_override(type_, namespace):
        type_.__module__ = f"gi.repository.{namespace}"
        type_.__overridden__ = type_

    fix_override(GLib.Idle, "GLib")
    fix_override(GLib.IOChannel, "GLib")
    fix_override(GLib.MainContext, "GLib")
    fix_override(GLib.MainLoop, "GLib")
    fix_override(GLib.PollFD, "GLib")
    fix_override(GLib.Source, "GLib")
    fix_override(GLib.Timeout, "GLib")
    fix_override(GLib.Variant, "GLib")
    fix_override(GObject.Binding, "GObject")
    fix_override(GObject.Object, "GObject")
    fix_override(GObject.Value, "GObject")


def is_classmethod(klass: type, name: str) -> bool:
    assert getattr(klass, name)
    for c in klass.__mro__:
        if name in c.__dict__:
            obj = c.__dict__.get(name)
            return isinstance(obj, (classmethod, staticmethod))
    return False


def is_ref_unref_copy_or_steal_function(name) -> bool:
    return bool(match(r"^(\w*_)?(ref|unref|copy|steal)(_\w*)?$", name))


def custom_docstring(subject: Callable | None) -> str | None:
    if subject.__doc__:
        doc = prepare_docstring(getdoc(subject))
        return (
            None if not doc or doc[0] == ":Constructors:" or match(r"^\w+\(.*\)", doc[0]) else "\n".join(doc)
        )

    try:
        key = _override_key(subject)
    except AttributeError:
        return None

    if key and (fun := getattr(overrides, key, None)) and (doc := getdoc(fun)):
        return "\n".join(prepare_docstring(doc))
    return None


def signature(subject: Callable, bound=False, is_async=False) -> Signature:
    if fun := getattr(overrides, _override_key(subject), None):
        return sphinx_signature(fun)

    return (
        async_signature(subject, bound=False) if is_async else sphinx_signature(subject, bound_method=bound)
    )


def async_signature(subject: Callable, bound=False) -> Signature:
    sig = sphinx_signature(subject, bound_method=bound)
    params = list(sig.parameters.values())[:-3]
    finish_sig = sphinx_signature(subject.get_finish_func())  # type: ignore[attr-defined]
    return Signature(params, return_annotation=finish_sig.return_annotation)


def vfunc_signature(subject: Callable) -> Signature:
    sig = signature(subject)
    return sig.replace(parameters=list(sig.parameters.values())[2:])


def _override_key(subject):
    if hasattr(subject, "__module__"):
        return f"{subject.__module__}_{subject.__name__}".replace(".", "_")
    elif ocls := getattr(subject, "__objclass__", None):
        return f"{ocls.__module__}_{ocls.__name__}_{subject.__name__}".replace(".", "_")

    return None
