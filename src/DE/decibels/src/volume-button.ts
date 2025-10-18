import Adw from "gi://Adw";
import GObject from "gi://GObject";

export class APVolumeButton extends Adw.Bin {
  volume!: number;
  muted!: boolean;

  static {
    GObject.registerClass(
      {
        GTypeName: "APVolumeButton",
        Template: "resource:///org/gnome/Decibels/volume-button.ui",
        InternalChildren: ["adjustment", "menu_button", "mute_button"],
        Properties: {
          volume: GObject.param_spec_double(
            "volume",
            "volume",
            "The current volume, regardless of mute status",
            0,
            1.0,
            0.5,
            GObject.ParamFlags.READWRITE,
          ),
          muted: GObject.param_spec_boolean(
            "muted",
            "muted",
            "Whether the audio is currently muted",
            false,
            GObject.ParamFlags.READWRITE,
          ),
        },
      },
      this,
    );
  }

  private mute_button_icon_cb(_widget: this, muted: boolean): string {
    return muted ? "audio-volume-muted-symbolic" : "audio-volume-high-symbolic";
  }

  private menu_button_icon_cb(
    _widget: this,
    volume: number,
    muted: boolean,
  ): string {
    if (muted || volume === 0) return "audio-volume-muted-symbolic";
    else if (volume === 1) return "audio-volume-high-symbolic";
    else if (volume > 0.5) return "audio-volume-medium-symbolic";
    else return "audio-volume-low-symbolic";
  }

  private tooltip_text_cb(
    _widget: this,
    volume: number,
    muted: boolean,
  ): string {
    if (muted || volume === 0) return _("Muted");
    else if (volume === 1) return _("Full Volume");
    else
      return imports.format.vprintf(
        /* Translators: this is the percentage of the current volume,
         * as used in the tooltip, eg. "49 %".
         * Translate the "%d" to "%Id" if you want to use localised digits,
         * or otherwise translate the "%d" to "%d".
         */
        C_("volume percentage", "%d %%"),
        [Math.round(100 * volume).toString()],
      );
  }
}
