from types import MethodType


from pygobject_docs.category import Category
from pygobject_docs.generate import (
    import_module,
    generate,
    generate_classes,
    generate_functions,
)


def test_generate_glib_functions(tmp_path):
    generate_functions("GLib", "2.0", tmp_path)

    assert (tmp_path / "functions.rst").exists()
    assert ".. deprecated" in (tmp_path / "functions.rst").read_text()


def test_generate_gobject_functions(tmp_path):
    generate_functions("GObject", "2.0", tmp_path)

    assert (tmp_path / "functions.rst").exists()
    assert ".. deprecated" in (tmp_path / "functions.rst").read_text()


def test_generate_classes(tmp_path):
    generate_classes("GLib", "2.0", tmp_path, Category.Classes)

    assert (tmp_path / "classes.rst").exists()
    assert (tmp_path / "class-Pid.rst").exists()


def test_generate_gobject(tmp_path):
    generate("GObject", "2.0", tmp_path)

    result_path = tmp_path / "GObject-2.0"
    assert (result_path / "functions.rst").exists()
    assert (result_path / "classes.rst").exists()
    assert (result_path / "constants.rst").exists()
    assert (result_path / "enums.rst").exists()
    assert (result_path / "functions.rst").exists()
    assert (result_path / "index.rst").exists()
    assert (result_path / "interfaces.rst").exists()
    assert (result_path / "structures.rst").exists()
    assert (result_path / "unions.rst").exists()
    assert (result_path / "class-Object.rst").exists()


def test_gi_method_type():
    gobject = import_module("GObject", "2.0")

    assert not callable(gobject.ParamSpecBoxed.name)
    assert callable(gobject.ParamSpecBoxed.do_values_cmp)
    assert type(gobject.ParamSpecBoxed.do_values_cmp) is MethodType
