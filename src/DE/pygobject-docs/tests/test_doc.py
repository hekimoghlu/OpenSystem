from textwrap import dedent

import pytest

from pygobject_docs.doc import rstify
from pygobject_docs.gir import load_gir_file


@pytest.fixture
def glib():
    return load_gir_file("GLib", "2.0")


@pytest.mark.parametrize(
    "text, expected",
    [
        ["`class`", "``class``"],
        ["`char*[]`", "``char*[]``"],
        ["`func()`", "``func()``"],
        [
            "`gtk_list_store_insert_with_values (list_store, iter, position...)`",
            "``gtk_list_store_insert_with_values (list_store, iter, position...)``",
        ],
        ["`cls::prop`", "``cls::prop``"],
        ["`interface`s", "``interface``\\s"],
        ["test `Class` and `Interface`s.", "test ``Class`` and ``Interface``\\s."],
        [
            "[`DBusActivatable` interface](https://some-url#dbus)",
            "```DBusActivatable`` interface <https://some-url#dbus>`__",
        ],
        [
            "A link [https://uefi.org/pnp_id_list](https://uefi.org/pnp_id_list).",
            "A link `https://uefi.org/pnp_id_list <https://uefi.org/pnp_id_list>`__\\.",
        ],
        [
            "You can check which `GdkGLContext` is the current one by using [func@Gdk.GLContext.get_current]; lorum",
            "You can check which ``GdkGLContext`` is the current one by using :obj:`~gi.repository.Gdk.GLContext.get_current`\\; lorum",
        ],
        [
            "[func@Gdk.GLContext.get_current]; [func@Gdk.GLContext.clear_current].",
            ":obj:`~gi.repository.Gdk.GLContext.get_current`\\; :obj:`~gi.repository.Gdk.GLContext.clear_current`\\.",
        ],
        ["The toplevel element is `<interface>`.", "The toplevel element is ``<interface>``\\."],
    ],
)
def test_markdown_inline_code(glib, text, expected):
    rst = rstify(text, gir=glib)

    assert rst == expected


def test_convert_constant(glib):
    text = "Lorem %TRUE ipsum %FALSE %NULL."

    rst = rstify(text, gir=glib)

    assert rst == "Lorem :const:`True` ipsum :const:`False` :const:`None`."


def test_convert_c_constant(glib):
    text = "Splits a %G_SEARCHPATH_SEPARATOR-separated list of files."

    rst = rstify(text, gir=glib)

    assert ":const:`~gi.repository.GLib.SEARCHPATH_SEPARATOR`" in rst


def test_convert_markdown_link(glib):
    text = """Lorem ipsum [link text 1](https://gitlab.gnome.org/some_url).
    More Lorem ipsum [second link](https://gitlab.gnome.org/second_url)."""

    rst = rstify(text, gir=glib)

    assert "`link text 1 <https://gitlab.gnome.org/some_url>`" in rst
    assert "`second link <https://gitlab.gnome.org/second_url>`" in rst


def test_convert_gtk_doc_code_snippet(glib):
    text = dedent(
        """\
    Lorem ipsum

    |[<!-- language="C" -->
      char buf[G_ASCII_DTOSTR_BUF_SIZE];

      fprintf (out, "value=%s\n", g_ascii_dtostr (buf, sizeof (buf), value));
    ]|
    """
    )

    rst = rstify(text, gir=glib)

    assert ".. code-block:: C" in rst
    assert "   char " in rst
    assert "]|" not in rst


def test_convert_gtk_doc_code_snippet_without_extra_line(glib):
    text = dedent(
        """\
    Lorem ipsum
    |[<!-- language="C" -->
      char buf[G_ASCII_DTOSTR_BUF_SIZE];

      fprintf (out, "value=%s\n", g_ascii_dtostr (buf, sizeof (buf), value));
    ]|
    """
    )

    rst = rstify(text, gir=glib)

    assert ".. code-block:: C" in rst
    assert "   char " in rst
    assert "]|" not in rst


def test_convert_markdown_code_snippet(glib):
    text = dedent(
        """\
    Lorem ipsum

    ```c
      char buf[G_ASCII_DTOSTR_BUF_SIZE];

      fprintf (out, "value=%s\n", g_ascii_dtostr (buf, sizeof (buf), value));
    ```
    """
    )

    rst = rstify(text, gir=glib)

    assert ".. code-block:: c" in rst
    assert "   char " in rst
    assert "```" not in rst


def test_convert_markdown_code_snippet_without_extra_line(glib):
    text = dedent(
        """\
    Lorem ipsum
    ```c
      char buf[G_ASCII_DTOSTR_BUF_SIZE];

      fprintf (out, "value=%s\n", g_ascii_dtostr (buf, sizeof (buf), value));
    ```
    """
    )

    rst = rstify(text, gir=glib)

    assert ".. code-block:: c" in rst
    assert "   char " in rst
    assert "```" not in rst


def test_convert_xml_code_block(glib):
    text = dedent(
        """\
        ```xml
        <?xml version="1.0" encoding="UTF-8">
        <interface domain="your-app">
        ...
        </interface>
        ```

        Lorum ipsum
        """
    )

    expected = dedent(
        """\
        .. code-block:: xml
            :dedent:

            <?xml version="1.0" encoding="UTF-8">
            <interface domain="your-app">
            ...
            </interface>

        Lorum ipsum"""
    )

    rst = rstify(text, gir=glib)

    assert dedent(rst) == expected


def test_convert_css_code_block(glib):
    text = dedent(
        """\
        # CSS nodes

        |[<!-- language="plain" -->
        list[.separators][.rich-list][.navigation-sidebar][.boxed-list]
        ╰── row[.activatable]
        ]|

        `GtkListBox` uses a single CSS node named list.
        """
    )

    expected = dedent(
        """\
        CSS nodes
        --------------------------------------------------------------------------------


        .. code-block:: plain
            :dedent:

            list[.separators][.rich-list][.navigation-sidebar][.boxed-list]
            ╰── row[.activatable]

        ``GtkListBox`` uses a single CSS node named list."""
    )

    rst = rstify(text, gir=glib)

    assert dedent(rst) == expected


def test_class_link(glib):
    text = "Lorem ipsum [class@Gtk.Builder] et amilet"

    rst = rstify(text, gir=glib)

    assert ":obj:`~gi.repository.Gtk.Builder`" in rst


def test_class_link_without_namespace(glib):
    text = "Lorem ipsum [class@SomeClass] et amilet"

    rst = rstify(text, gir=glib)

    assert ":obj:`~gi.repository.GLib.SomeClass`" in rst


def test_method_link(glib):
    text = "Lorem ipsum [method@Gtk.Builder.foo] et amilet"

    rst = rstify(text, gir=glib)

    assert ":obj:`~gi.repository.Gtk.Builder.foo`" in rst


def test_property_link(glib):
    text = "Lorem ipsum [property@Foo.TestClass:property-name] et amilet"

    rst = rstify(text, gir=glib)

    assert ":obj:`~gi.repository.Foo.TestClass.props.property_name`" in rst


def test_multiple_property_link(glib):
    text = "Lorem ipsum [property@TestClass:property-name] et [property@Foo.SomeClass:second-name]"

    rst = rstify(text, gir=glib)

    assert ":obj:`~gi.repository.GLib.TestClass.props.property_name`" in rst
    assert ":obj:`~gi.repository.Foo.SomeClass.props.second_name`" in rst


def test_property_link_without_namespace(glib):
    text = "Lorem ipsum [property@TestClass:property-name] et amilet"

    rst = rstify(text, gir=glib)

    assert ":obj:`~gi.repository.GLib.TestClass.props.property_name`" in rst


def test_signal_link(glib):
    text = "Lorem ipsum [signal@Foo.TestClass::signal-name] et amilet"

    rst = rstify(text, gir=glib)

    assert ":obj:`~gi.repository.Foo.TestClass.signals.signal_name`" in rst


def test_signal_link_without_namespace(glib):
    text = "Lorem ipsum [signal@TestClass::signal-name] et amilet"

    rst = rstify(text, gir=glib)

    assert ":obj:`~gi.repository.GLib.TestClass.signals.signal_name`" in rst


def test_paragraph(glib):
    text = "Lorem ipsum\n\net amilet"

    rst = rstify(text, gir=glib)

    assert rst == "Lorem ipsum\n\net amilet"


def test_parameters(glib):
    text = "Lorem @ipsum et amilet"

    rst = rstify(text, gir=glib)

    assert rst == "Lorem ``ipsum`` et amilet"


def test_parameter_remove_pointer(glib):
    text = "Lorem *@ipsum et amilet"

    rst = rstify(text, gir=glib)

    assert rst == "Lorem ``ipsum`` et amilet"


def test_italic_text(glib):
    text = "This is a func_name and _italic text_."

    rst = rstify(text, gir=glib)

    assert rst == "This is a func_name and *italic text*\\."


def test_keyboard_shortcut(glib):
    text = "by pressing <kbd>Escape</kbd> or"

    rst = rstify(text, gir=glib)

    assert ":kbd:`Escape`" in rst


def test_combined_keyboard_shortcut(glib):
    text = "by pressing <kbd>Alt</kbd>+<kbd>w</kbd> or"

    rst = rstify(text, gir=glib)

    assert ":kbd:`Alt`" in rst
    assert ":kbd:`w`" in rst


def test_combined_keyboard_shortcut_with_space(glib):
    text = "by pressing <kbd>Alt</kbd>+<kbd>Page Down</kbd> or"

    rst = rstify(text, gir=glib)

    assert ":kbd:`Alt`" in rst
    assert ":kbd:`Page Down`" in rst


def test_combined_keyboard_shortcut_with_unicode_arrow(glib):
    text = "by pressing <kbd>Alt</kbd>+<kbd>↓</kbd> or"

    rst = rstify(text, gir=glib)

    assert ":kbd:`Alt`" in rst
    assert ":kbd:`↓`" in rst


def test_code_abbreviation(glib):
    text = "This is a func_name_ and _italic text_."

    rst = rstify(text, gir=glib)

    assert rst == "This is a ``func_name_`` and *italic text*\\."


def test_code_abbreviation_with_ellipsis(glib):
    text = "the g_convert_… functions"

    rst = rstify(text, gir=glib)

    assert rst == "the ``g_convert_``\\… functions"


def test_whitespace_before_lists(glib):
    text = dedent(
        """\
        line of text.

        - list item.
        """
    )

    rst = rstify(text, gir=glib)

    assert rst == dedent(
        """\
        line of text.

        - list item."""
    )


def test_multi_line_list_item(glib):
    text = dedent(
        """\
        - line one
          line two
        """
    )

    rst = rstify(text, gir=glib)

    assert rst == dedent(
        """\
        - line one
          line two"""
    )


def test_multi_line_list_item_with_paragraphs(glib):
    text = dedent(
        """\
        - item one
          line two

          paragraph two
        - item two
        """
    )

    rst = rstify(text, gir=glib)

    assert rst == dedent(
        """\
        - item one
          line two


        paragraph two
        - item two"""
    )


def test_simple_table(glib):
    text = dedent(
        """\
        | field 1 | field 2 |
        | field 3 | long field 4 |

        Lorum ipsum
        """
    )

    rst = rstify(text, gir=glib)

    assert rst == dedent(
        """\
        .. list-table::

            * - field 1
              - field 2
            * - field 3
              - long field 4

        Lorum ipsum"""
    )


def test_table_with_header_row(glib):
    text = dedent(
        """\
        | header 1 | header 2     |
        | -        | ---          |
        | field 1  | field 2      |
        | field 3  | long field 4 |

        """
    )

    rst = rstify(text, gir=glib)

    assert rst == dedent(
        """\
        .. list-table::
            :header-rows: 1

            * - header 1
              - header 2
            * - field 1
              - field 2
            * - field 3
              - long field 4"""
    )


def test_table_with_solid_header_row(glib):
    text = dedent(
        """\
        | header 1 | header 2     |
        |----------|--------------|
        | field 1  | field 2      |
        | field 3  | long field 4 |

        """
    )

    rst = rstify(text, gir=glib)

    assert rst == dedent(
        """\
        .. list-table::
            :header-rows: 1

            * - header 1
              - header 2
            * - field 1
              - field 2
            * - field 3
              - long field 4"""
    )


def test_table_with_multiline_content(glib):
    text = dedent(
        """\
        | | | | |
        | --- | --- | ---- | --- |
        | "none" | ![](default.png) "default" | ![](help.png) "help" |
        | ![](pointer.png) "pointer" | ![](cell_cursor.png) "cell" |

        Lorum ipsum
        """
    )

    rst = rstify(text, gir=glib, image_base_url="http://example.com")

    # +-------------------------------------------+-----------------------------------------------+
    # |                                           |                                               |
    # +===========================================+===============================================+
    # | "none"                                    | .. image:: http://example.com/default.png     |
    # |                                           | "default"                                     |
    # +-------------------------------------------+-----------------------------------------------+
    # | .. image:: http://example.com/pointer.png | .. image:: http://example.com/cell_cursor.png |
    # | "pointer"                                 | "cell"                                        |
    # +-------------------------------------------+-----------------------------------------------+

    # Dedent the rst output, to shrink lines that only contain spaces.
    assert dedent(rst) == dedent(
        """\
        .. list-table::
            :header-rows: 1

            * -
              -
              -
              -
            * - "none"
              - .. image:: http://example.com/default.png

                "default"
              - .. image:: http://example.com/help.png

                "help"
            * - .. image:: http://example.com/pointer.png

                "pointer"
              - .. image:: http://example.com/cell_cursor.png

                "cell"

        Lorum ipsum"""
    )


def test_remove_tags(glib):
    text = "I/O Priority # {#io-priority}"

    rst = rstify(text, gir=glib)

    assert rst == "I/O Priority"


@pytest.mark.parametrize(
    "text, expected",
    [
        ["This is a #GQueue", "This is a :obj:`~gi.repository.GLib.Queue`"],
        ["a #gint32 value", "a :obj:`int` value"],
        ["#gint32 value", ":obj:`int` value"],
        [
            "In a url <http://example.com#section-123>",
            "In a url `http://example.com#section-123 <http://example.com#section-123>`__",
        ],
        [
            "If we were to use g_variant_get_child_value()",
            "If we were to use :func:`~gi.repository.GLib.Variant.get_child_value`",
        ],
        ["Good old function g_access()", "Good old function :func:`~gi.repository.GLib.access`"],
        [r"%G_SPAWN_ERROR_TOO_BIG", ":const:`~gi.repository.GLib.SpawnError.TOO_BIG`"],
        ["A function_with_*() function", "A ``function_with_*()`` function"],
    ],
)
def test_c_symbol_to_python(glib, text, expected):
    rst = rstify(text, gir=glib)

    assert rst == expected


def test_html_picture_tag(glib):
    text = dedent(
        """\
    Freeform text.

    <picture>
        <source srcset="application-window-dark.png" media="(prefers-color-scheme: dark)">
        <img src="application-window.png" alt="application-window">
    </picture>

    More freeform text.
    """
    )

    rst = rstify(text, gir=glib, image_base_url="https://example.com")

    assert "Freeform text." in rst
    assert "More freeform text." in rst
    assert ".. image:: https://example.com/application-window.png" in rst


def test_docbook_note(glib):
    text = "<note>This is a it</note>"

    rst = rstify(text, gir=glib)

    assert "This is a it" in rst
    assert "note" not in rst


def test_docbook_literal(glib):
    text = "The regex will be compiled using <literal>PCRE2_UTF</literal>."

    rst = rstify(text, gir=glib)

    assert "``PCRE2_UTF``\\." in rst
    assert "literal" not in rst


def test_escape_asterisk(glib):
    text = "- A *pointer."

    rst = rstify(text, gir=glib)

    assert rst == "- A \\*pointer."
