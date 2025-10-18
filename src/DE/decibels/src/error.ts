import Adw from "gi://Adw";
import GLib from "gi://GLib";
import GObject from "gi://GObject";

import { APHeaderBar } from "./header.js";

export class APErrorState extends Adw.Bin {
  private _statusPage!: Adw.StatusPage;

  headerbar!: APHeaderBar;

  static {
    GObject.registerClass(
      {
        GTypeName: "APErrorState",
        Template: "resource:///org/gnome/Decibels/error.ui",
        InternalChildren: ["statusPage"],
        Children: ["headerbar"],
      },
      this,
    );
  }

  constructor(params?: Partial<Adw.Bin.ConstructorProperties>) {
    super(params);
  }

  private show_message(message: string) {
    this._statusPage.set_description(message);
  }

  show_error(title: string, error: unknown) {
    this._statusPage.title = title;

    if (error instanceof Error) {
      this.show_message(error.message);
    } else if (error instanceof GLib.Error) {
      this.show_message(error.message);
    } else {
      console.error("error: ", error);
      this.show_message(
        error ? (error as Error).toString() : _("An unknown error happened"),
      );
    }
  }
}
