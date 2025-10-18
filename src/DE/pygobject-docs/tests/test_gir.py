import pytest

from pygobject_docs import gir as _gir


@pytest.fixture
def glib():
    g = _gir.load_gir_file("GLib", "2.0")
    assert g
    return g


@pytest.fixture
def gobject():
    g = _gir.load_gir_file("GObject", "2.0")
    assert g
    return g


def test_gir_dirs():
    dirs = _gir.gir_dirs()

    assert dirs


def test_gir_namespace(glib):
    assert glib.namespace == ("GLib", "2.0")


def test_gir_dependencies(gobject):
    dep = next(gobject.dependencies)

    assert dep == "GLib-2.0"


def test_gir_function_doc(glib):
    doc = glib.doc("filename_from_utf8")

    assert doc
    assert doc.startswith("Converts ")


def test_gir_class_doc(gobject):
    doc = gobject.doc("Object")

    assert doc
    assert doc.startswith("The base object type.")


def test_gir_function_parameter_docs(gobject):
    doc = gobject.parameter_doc("boxed_copy", "boxed_type")

    assert doc


def test_gir_function_return_doc(gobject):
    doc = gobject.return_doc("boxed_copy")

    assert doc


def test_gir_deprecated(glib):
    ver, doc = glib.deprecated("basename")

    assert ver == "2.2"
    assert doc


def test_gir_since(glib):
    ver = glib.since("atomic_int_add")

    assert ver == "2.4"


def test_virtual_method(gobject):
    doc = gobject.member_doc("virtual-method", "Object", "notify")

    assert doc
    assert doc.startswith("Emits ")


def test_method_parameter_docs(gobject):
    doc = gobject.member_parameter_doc("method", "Object", "notify", "property_name")

    assert doc


def test_class_method_parameter_docs(gobject):
    doc = gobject.member_parameter_doc("method", "Object", "find_property", "property_name")

    assert doc


def test_virtual_method_parameter_docs(gobject):
    doc = gobject.member_parameter_doc("virtual-method", "ParamSpec", "values_cmp", "value1")

    assert doc == ""


def test_signal_docs(gobject):
    doc = gobject.member_doc("signal", "Object", "notify")

    assert doc


def test_deprecated_method(gobject):
    version, message = gobject.member_deprecated("method", "Binding", "get_target")

    assert version == "2.68"
    assert "Use g_binding_dup_target()" in message


def test_struct_for(gobject):
    obj = gobject.struct_for("ObjectClass")

    assert obj == "Object"


def test_enum_field(glib):
    doc = glib.member_doc("field", "IOFlags", "APPEND".lower())

    assert doc
