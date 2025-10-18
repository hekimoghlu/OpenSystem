import Adw from "gi://Adw";
import Gio from "gi://Gio";
import GObject from "gi://GObject";
import Gtk from "gi://Gtk?version=4.0";

import { MPRIS } from "./mpris.js";
import { AddActionEntries, Window } from "./window.js";
import { APMediaStream } from "./stream.js";

export class Application extends Adw.Application {
  static {
    GObject.registerClass(this);
  }

  mpris: MPRIS;

  constructor() {
    super({
      application_id: pkg.name,
      resource_base_path: "/org/gnome/Decibels",
      flags: Gio.ApplicationFlags.HANDLES_OPEN,
    });

    this.mpris = new MPRIS();

    Gtk.Window.set_default_icon_name(pkg.name);

    this.connect(
      "notify::active-window",
      this.active_window_changed_cb.bind(this),
    );

    (this.add_action_entries as AddActionEntries)([
      {
        name: "new-window",
        activate: () => {
          this.present_new_window();
        },
      },
      {
        name: "quit",
        activate: () => {
          this.quit();
        },
      },
      {
        name: "about",
        activate: () => {
          this.show_about_dialog_cb();
        },
      },
    ]);

    this.set_accels_for_action("app.new-window", ["<Control>n"]);
    this.set_accels_for_action("app.quit", ["<Control>q"]);
    this.set_accels_for_action("win.open-file", ["<Control>o"]);
  }

  private show_about_dialog_cb() {
    const aboutDialog = Adw.AboutDialog.new_from_appdata(
      `/org/gnome/Decibels/${pkg.name}.metainfo.xml`,
      // remove commit tag
      pkg.version.split("-")[0],
    );
    aboutDialog.set_version(pkg.version);
    aboutDialog.set_developers([
      "Angelo Verlain https://vixalien.com",
      "David Keller https://gitlab.com/BlobCodes",
    ]);
    aboutDialog.set_artists(["kramo https://kramo.page"]);
    aboutDialog.set_designers(["Allan Day"]);
    /* Translators: Replace "translator-credits" with your names, one name per line */
    aboutDialog.set_translator_credits(_("translator-credits"));

    aboutDialog.present(this.get_active_window());
  }

  private present_new_window() {
    const window = new Window({ application: this });
    if (pkg.name.endsWith("Devel")) window.add_css_class("devel");
    window.present();

    return window;
  }

  private new_stream_listener: [APMediaStream, number] | null = null;

  private cleanup_stream_listener() {
    if (!this.new_stream_listener) return;

    const [stream, listener_id] = this.new_stream_listener;
    stream.disconnect(listener_id);
    this.new_stream_listener = null;
  }

  private switch_mpris_stream(stream: APMediaStream) {
    this.mpris.switch_stream(stream);
    if (!this.mpris.started) this.mpris.start();
    this.cleanup_stream_listener();
  }

  private active_window_changed_cb() {
    const window = this.active_window;

    if (!(window instanceof Window) || !window.stream) return;

    const stream = window.stream;

    if (stream.media_info) {
      this.switch_mpris_stream(stream);
    } else {
      // this stream has not yet loaded a track. switch to it as soon as it
      // loads a track to avoid showing "Unknown File" as playing
      this.new_stream_listener = [
        stream,
        stream.connect("notify::media-info", () => {
          this.switch_mpris_stream(stream);
        }),
      ];
    }
  }

  vfunc_activate(): void {
    this.present_new_window();
  }

  vfunc_open(files: Gio.FilePrototype[]): void {
    const is_single_file = files.length === 1,
      window = this.get_active_window() as Window;

    if (is_single_file && window && !window.stream?.media_info) {
      // we are opening a single file, and the current window has no file open,
      // so open the file in the window
      void window.load_file(files[0]);
      return;
    }

    for (const file of files) {
      const window = this.present_new_window();
      // only autoplay when only one file is opened
      void window.load_file(file, is_single_file);
    }
  }
}
