#!/usr/bin/env gjs

const GLib = imports.gi.GLib;
const Gtk = imports.gi.Gtk;
const GtkSource = imports.gi.GtkSource;
const Spelling = imports.gi.Spelling;

Gtk.init();
GtkSource.init();
Spelling.init();

let mainLoop = GLib.MainLoop.new(null, false);
let win = Gtk.Window.new();
let scroller = Gtk.ScrolledWindow.new();
let view = new GtkSource.View();
let buffer = view.get_buffer();
let checker = Spelling.Checker.get_default();
let adapter = Spelling.TextBufferAdapter.new(buffer, checker);

win.set_child(scroller);
scroller.set_child(view);
buffer.set_style_scheme(GtkSource.StyleSchemeManager.get_default().get_scheme('Adwaita'));

view.insert_action_group('spelling', adapter);
view.set_extra_menu(adapter.get_menu_model());

adapter.set_enabled(true)

win.connect('close-request', function() {
    mainLoop.quit();
});

win.present();

mainLoop.run();
