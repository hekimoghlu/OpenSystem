import pytest
from gi.repository import GLib, Gio, GObject

from pygobject_docs.inspect import (
    async_signature,
    custom_docstring,
    is_classmethod,
    is_ref_unref_copy_or_steal_function,
    signature,
)


def test_function_signature():
    def func(arg: str, obj: GObject.Object) -> str | int:
        return 1

    assert str(signature(func)) == "(arg: str, obj: ~gi.repository.GObject.Object) -> str | int"


def test_builtin_function_signature():
    assert (
        str(signature(GObject.add_emission_hook))
        == "(type: ~gi.repository.GObject.Object, name: str, callback: ~typing.Callable[[...], None], *args: ~typing.Any) -> None"
    )


@pytest.mark.skip(reason="Inconsistent results on my laptop and Gitlab")
def test_function_with_callback_signature():
    assert (
        str(signature(GObject.signal_add_emission_hook))
        == "(signal_id: int, detail: int, hook_func: ~typing.Callable[[~gi.repository.GObject.SignalInvocationHint, int, ~typing.Sequence[~typing.Any]], bool], data_destroy: ~typing.Callable[[], None]) -> int"
    )


def test_class_signature():
    class Foo:
        def method(self, arg: int) -> GObject.Object:
            ...

    assert str(signature(Foo.method)) == "(self, arg: int) -> ~gi.repository.GObject.Object"


def test_gi_function_signature():
    assert (
        str(signature(GObject.flags_complete_type_info))
        == "(g_flags_type: ~gobject.GType, const_values: ~gi.repository.GObject.FlagsValue) -> ~gi.repository.GObject.TypeInfo"
    )
    assert (
        str(signature(GObject.signal_handler_unblock))
        == "(instance: ~gi.repository.GObject.Object, handler_id: int) -> None"
    )


def test_builtin_method():
    assert (
        str(signature(GObject.GObject.bind_property))
        == "(source_property: str, target: ~gi.repository.GObject.Object, target_property: str, flags: ~gi.repository.GObject.BindingFlags | None = 0, transform_to: ~typing.Callable[[~gi.repository.GObject.Binding, ~typing.Any, ~typing.Any], ~typing.Any] | None = None, transform_from: ~typing.Callable[[~gi.repository.GObject.Binding, ~typing.Any, ~typing.Any], ~typing.Any] | None = None, user_data: ~typing.Any = None) -> ~gi.repository.GObject.Binding"
    )


@pytest.mark.xfail(reason="default argument values not implemented")
def test_function_with_default_value():
    assert str(signature(GLib.base64_encode)) == "(data: ~typing.Sequence[str] = None) -> str"


@pytest.mark.desktop
def test_method_with_multiple_return_values():
    from gi.repository import Gtk

    assert str(signature(Gtk.Scrollable.get_border)) == "(self) -> tuple[bool, ~gi.repository.Gtk.Border]"


def test_python_method_is_classmethod():
    class A:
        @classmethod
        def yup(cls):
            ...

        def nope(self):
            ...

    assert is_classmethod(A, "yup")
    assert not is_classmethod(A, "nope")


def test_gi_function_is_classmethod():
    assert is_classmethod(GObject.Object, "install_properties")
    assert not is_classmethod(GObject.Object, "notify")
    assert not is_classmethod(GObject.ValueArray, "append")


def test_async_signature():
    sig = async_signature(Gio.File.append_to_async)

    assert (
        str(sig)
        == "(self, flags: ~gi.repository.Gio.FileCreateFlags, io_priority: int) -> ~gi.repository.Gio.FileOutputStream"
    )


def test_method_with_length_parameter():
    # C signature is like: GArray* (..., int *length)
    sig = signature(GObject.type_children)

    assert str(sig) == "(type: ~gobject.GType) -> list[~gobject.GType]"


def test_custom_docstring_from_custom_overrides():
    assert ":param detailed_signal:" in custom_docstring(GObject.Object.connect)


def test_custom_docstring_from_gi_overrides():
    assert ":returns:" in custom_docstring(GObject.Object.freeze_notify)


def test_custom_docstring_from_class():
    doc = custom_docstring(GLib.Error)

    assert doc
    assert "attribute::" in doc


def test_skip_signature_docstring_overrides():
    assert not custom_docstring(GLib.io_add_watch)
    assert not custom_docstring(GLib.child_watch_add)


@pytest.mark.parametrize("fragment", ["ref", "unref", "copy", "steal"])
def test_hide_ref_count_functions(fragment):
    assert is_ref_unref_copy_or_steal_function(fragment)
    assert is_ref_unref_copy_or_steal_function(f"{fragment}_me")
    assert is_ref_unref_copy_or_steal_function(f"some_{fragment}")
    assert is_ref_unref_copy_or_steal_function(f"some_{fragment}_embedded")


def test_not_hide_ref_count_functions():
    assert not is_ref_unref_copy_or_steal_function("dounref")
    assert not is_ref_unref_copy_or_steal_function("somunref_function")
    assert not is_ref_unref_copy_or_steal_function("stealler_embedded")


def test_function_with_closure():
    sig = signature(GObject.signal_handler_find)

    assert sig


def test_gio_file_stream_seek_vfunc():
    signature(Gio.FileIOStream.do_seek)
