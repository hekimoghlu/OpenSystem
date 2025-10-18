/*
 * Copyright 2013 Meg Ford
             2020 Kavan Mevada
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 *  Author: Meg Ford <megford@gnome.org>
 *          Kavan Mevada <kavanmevada@gmail.com>
 *
 */

// based on code from GNOME Sound Recorder

import GLib from "gi://GLib";
import GObject from "gi://GObject";
import Gst from "gi://Gst";

export class APWaveformGenerator extends GObject.Object {
  loaded_peaks: number[] = [];

  static {
    GObject.registerClass(
      {
        GTypeName: "APWaveformGenerator",
        Properties: {
          peaks: GObject.param_spec_boxed(
            "peaks",
            "Peaks",
            "The peaks of the currently playing song",
            Object.$gtype,
            GObject.ParamFlags.READABLE,
          ),
        },
      },
      this,
    );
  }

  private _peaks: number[] = [];
  private pipeline?: Gst.Bin;
  private bus?: Gst.Bus;
  private callback_id?: number;

  get peaks() {
    return this._peaks;
  }

  set peaks(peaks: number[]) {
    this._peaks = peaks;
    this.notify("peaks");
  }

  constructor() {
    super();
  }

  restart() {
    if (this.callback_id && this.bus) {
      this.bus.disconnect(this.callback_id);
      this.bus.remove_signal_watch();
    }

    this.pipeline?.set_state(Gst.State.NULL);

    this.loaded_peaks.length = 0;
    this.peaks.length = 0;
    this.notify("peaks");
  }

  generate_peaks_async(uri: string): void {
    this.pipeline = Gst.parse_launch(
      `uridecodebin name=uridecodebin ! audioconvert ! audio/x-raw,channels=1 ! level name=level interval=${this.INTERVAL} ! fakesink name=faked`,
    ) as Gst.Bin;

    const fakesink = this.pipeline.get_by_name("faked");
    fakesink?.set_property("qos", false);
    fakesink?.set_property("sync", false);

    const uridecodebin = this.pipeline.get_by_name("uridecodebin");
    uridecodebin?.set_property("uri", uri);

    this.pipeline.set_state(Gst.State.PLAYING);

    const bus = this.pipeline.get_bus();
    if (!bus) return;

    this.bus = bus;
    bus.add_signal_watch_full(GLib.PRIORITY_DEFAULT_IDLE);
    this.callback_id = bus.connect(
      "message",
      (_bus: Gst.Bus, message: Gst.Message) => {
        switch (message.type) {
          case Gst.MessageType.ELEMENT: {
            const s = message.get_structure();
            if (s && s.has_name("level")) {
              const peakVal = s.get_value(
                "rms",
              ) as unknown as GObject.ValueArray;

              if (peakVal) {
                const peak = peakVal.get_nth(0) as number;
                this.loaded_peaks.push(Math.pow(10, (peak || 0) / 20));
              }
            }
            break;
          }
          case Gst.MessageType.EOS: {
            const highest_peak = Math.max(...this.loaded_peaks);
            const peak_coefficient = 1 / (highest_peak <= 0 ? 1 : highest_peak);
            this.peaks = this.loaded_peaks.map((it) => it * peak_coefficient);
            this.loaded_peaks.length = 0;

            this.pipeline?.set_state(Gst.State.NULL);
            break;
          }
        }
      },
    );
  }

  INTERVAL = 100000000;
}
