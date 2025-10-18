# libspelling

A spellcheck library for GTK 4.

This library is heavily based upon GNOME Text Editor and GNOME Builder's
spellcheck implementation. However, it is licensed LGPLv2.1+.

## Documentation

[Our documentation](https://gnome.pages.gitlab.gnome.org/libspelling/libspelling-1/) is updated on every commit.

## Installing Dictionaries

Currently, libspelling wraps `enchant-2` only.
That means to get additional dictionaries you need to follow the same directions as enchant.
Generally, that means installing packages like `apsell-en` or `hunspell-fr` and so forth.

Enchant should pick those up and use them the next time a libspelling-based application is run.

## Example

### In C

```c
SpellingChecker *checker = spelling_checker_get_default ();
g_autoptr(SpellingTextBufferAdapter) adapter = spelling_text_buffer_adapter_new (source_buffer, checker);
GMenuModel *extra_menu = spelling_text_buffer_adapter_get_menu_model (adapter);

gtk_text_view_set_extra_menu (GTK_TEXT_VIEW (source_view), extra_menu);
gtk_widget_insert_action_group (GTK_WIDGET (source_view), "spelling", G_ACTION_GROUP (adapter));
spelling_text_buffer_adapter_set_enabled (adapter, TRUE);
```

### In Python

```python
from gi.repository import Spelling

checker = Spelling.Checker.get_default()
adapter = Spelling.TextBufferAdapter.new(buffer, checker)
extra_menu = adapter.get_menu_model()

view.set_extra_menu(extra_menu)
view.insert_action_group('spelling', adapter)

adapter.set_enabled(True)
```

### In JavaScript

```js
const Spelling = imports.gi.Spelling;

let checker = Spelling.Checker.get_default()
let adapter = Spelling.TextBufferAdapter.new(buffer, checker)
let extra_menu = adapter.get_menu_model()

view.set_extra_menu(extra_menu)
view.insert_action_group('spelling', adapter)

adapter.set_enabled(true)
```

### In Rust

Add the [bindings dependency](https://crates.io/crates/libspelling) to your Cargo.toml

```rust
let checker = libspelling::Checker::default();
let adapter = libspelling::TextBufferAdapter::new(&buffer, &checker);
let extra_menu = adapter.menu_model();

view.set_extra_menu(Some(&extra_menu));
view.insert_action_group("spelling", Some(&adapter));

adapter.set_enabled(true);
```
