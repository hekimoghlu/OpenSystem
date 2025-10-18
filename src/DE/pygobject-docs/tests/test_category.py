import pytest

from pygobject_docs.category import (
    Category,
    determine_category,
    determine_member_category,
    MemberCategory,
)
from pygobject_docs.generate import import_module
from pygobject_docs.members import own_dir


@pytest.fixture
def glib():
    return import_module("GLib", "2.0")


@pytest.fixture
def gobject():
    return import_module("GObject", "2.0")


def test_determine_function(glib):
    category = determine_category(glib, "filename_from_utf8")

    assert category == Category.Functions


@pytest.mark.parametrize("name", ["Object", "GType", "Binding"])
def test_determine_gobject_class(gobject, name):
    category = determine_category(gobject, name)

    assert category == Category.Classes


def test_determine_glib_structure(glib):
    category = determine_category(glib, "Error")

    assert category == Category.Structures


@pytest.mark.parametrize("name", ["CClosure", "EnumClass", "Value"])
def test_determine_gobject_structure(gobject, name):
    category = determine_category(gobject, name)

    assert category == Category.Structures


def test_determine_glib_union(glib):
    category = determine_category(glib, "FloatIEEE754")

    assert category == Category.Unions


@pytest.mark.parametrize("name", ["GInterface", "TypePlugin"])
def test_determine_gobject_interface(gobject, name):
    category = determine_category(gobject, name)

    assert category == Category.Interfaces


@pytest.mark.parametrize("name", ["OptionFlags", "IOFlags"])
def test_determine_glib_flag(glib, name):
    category = determine_category(glib, name)

    assert category == Category.Enums


@pytest.mark.parametrize("name", ["GINT32_FORMAT", "BIG_ENDIAN"])
def test_determine_glib_constant(glib, name):
    category = determine_category(glib, name)

    assert category == Category.Constants


@pytest.mark.parametrize(
    "name,category",
    [
        ["threads_init", Category.Functions],
        ["unix_signal_add_full", Category.Functions],
    ],
)
def test_determine_glib_field_category(glib, name, category):
    actual_category = determine_category(glib, name)

    assert actual_category == category


@pytest.mark.parametrize("name,category", [["add_emission_hook", Category.Functions]])
def test_determine_gobject_field_category(gobject, name, category):
    actual_category = determine_category(gobject, name)

    assert actual_category == category


def test_all_glib_fields_are_categorized(glib):
    for name in dir(glib):
        determine_category(glib, name)


def test_all_gobject_fields_are_categorized(gobject):
    for name in dir(gobject):
        determine_category(gobject, name)


@pytest.mark.desktop
@pytest.mark.parametrize(
    "namespace,version",
    [
        ["Gtk", "4.0"],
        ["Gdk", "4.0"],
        ["Pango", "1.0"],
        ["HarfBuzz", "0.0"],
    ],
)
def test_all_gtk_fields_are_categorized(namespace, version):
    mod = import_module(namespace, version)

    for name in dir(mod):
        determine_category(mod, name)


def test_member_constructor(gobject):
    obj_type = gobject.Object
    category = determine_member_category(obj_type, "newv")

    assert category == MemberCategory.Constructors


@pytest.mark.parametrize(
    "name", ["connect", "connect_data", "find_property", "bind_property", "install_properties"]
)
def test_member_method(gobject, name):
    obj_type = gobject.Object
    category = determine_member_category(obj_type, name)

    assert category == MemberCategory.Methods


@pytest.mark.parametrize("name", ["do_get_property", "do_notify"])
def test_member_virtual_method(gobject, name):
    obj_type = gobject.Object
    category = determine_member_category(obj_type, name)

    assert category == MemberCategory.VirtualMethods


def test_enum_member(gobject):
    obj_type = gobject.BindingFlags
    category = determine_member_category(obj_type, "BIDIRECTIONAL")

    assert category == MemberCategory.Fields


@pytest.mark.parametrize("name", ["ref", "get_data"])
def test_member_ignored(gobject, name):
    obj_type = gobject.Object
    category = determine_member_category(obj_type, name)

    assert category == MemberCategory.Ignored


def test_all_glib_error_members_are_categorized(glib):
    obj_type = glib.Error

    for name in own_dir(obj_type):
        determine_member_category(obj_type, name)


def test_all_gobject_members_are_categorized(gobject):
    obj_type = gobject.Object

    for name in own_dir(obj_type):
        determine_member_category(obj_type, name)


def test_all_genum_members_are_categorized(gobject):
    obj_type = gobject.GEnum

    for name in own_dir(obj_type):
        determine_member_category(obj_type, name)


@pytest.mark.desktop
def test_gtk_member_fields():
    gtk = import_module("Gtk", "4.0")

    obj_type = gtk.Window

    for name in own_dir(obj_type):
        determine_member_category(obj_type, name)


def test_singular_name():
    assert Category.Functions.single == "function"
    assert Category.Interfaces.single == "interface"
    assert Category.Classes.single == "class"
    assert Category.Structures.single == "structure"
    assert Category.Unions.single == "union"
    assert Category.Enums.single == "enum"
    assert Category.Constants.single == "constant"
