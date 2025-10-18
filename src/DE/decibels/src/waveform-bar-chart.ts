import GObject from "gi://GObject";
import Gdk from "gi://Gdk?version=4.0";
import Gtk from "gi://Gtk?version=4.0";
import Graphene from "gi://Graphene";

/**
 * A {@link Gdk.Paintable} representing an audio waveform as a fixed-width, monochrome bar chart.
 *
 * Painting it results in a transparent image with equidistant bars of a fixed width.
 * The color, bar width and bar padding are configurable using GObject properties.
 *
 * If more bars need to be displayed than were supplied, linear interpolation is used to fill the gaps.
 * This algorithm is fast and results in a smooth waveform.
 *
 * If less bars need to be displayed than were supplied, the supplied data is mipmapped
 * (resolution is halved repeatedly) until the mipmap bias is reached, followed by linear interpolation.
 * While this method tends to lose fine details in the waveform,
 * it is simple, performant and minimizes artifacts when resizing the window.
 */
export class APWaveformBarChart
  extends GObject.Object
  implements Gdk.Paintable
{
  private _color: Gdk.RGBA = new Gdk.RGBA({
    red: 1,
    green: 1,
    blue: 1,
    alpha: 1,
  });
  private _bar_width!: number;
  private _bar_padding!: number;
  private peaks_mipmaps: number[][] = [];
  private _mipmap_bias!: number;

  static {
    GObject.registerClass(
      {
        GTypeName: "APWaveformBarChart",
        Properties: {
          color: GObject.param_spec_boxed(
            "color",
            "color",
            "The bar color",
            Gdk.RGBA.$gtype,
            GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
          ),
          bar_width: GObject.ParamSpec.float(
            "bar-width",
            "bar-width",
            "The width of the bars representing the audio level",
            GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
            0.0,
            100.0,
            2.0,
          ),
          bar_padding: GObject.ParamSpec.float(
            "bar-padding",
            "bar-padding",
            "The padding around each bar representing the audio level",
            GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
            0.0,
            100.0,
            1.0,
          ),
          peaks: GObject.param_spec_boxed(
            "peaks",
            "peaks",
            "The audio levels of the current track",
            Object.$gtype,
            GObject.ParamFlags.READWRITE,
          ),
          mipmap_bias: GObject.ParamSpec.float(
            "mipmap-bias",
            "mipmap-bias",
            "Bias controlling which mipmap level to use (higher = blurrier)",
            GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
            -32.0,
            32.0,
            0.25,
          ),
        },
        Implements: [Gdk.Paintable],
      },
      this,
    );
  }

  get color(): Gdk.RGBA {
    return this._color;
  }

  set color(color: Gdk.RGBA) {
    if (color == this._color) return;
    this._color = color;
    this.invalidate_contents();
    this.notify("color");
  }

  get bar_width(): number {
    return this._bar_width;
  }

  set bar_width(width: number) {
    if (width == this._bar_width) return;
    this._bar_width = width;
    this.invalidate_contents();
    this.notify("bar-width");
  }

  get bar_padding(): number {
    return this._bar_padding;
  }

  set bar_padding(padding: number) {
    if (padding == this._bar_padding) return;
    this._bar_padding = padding;
    this.invalidate_contents();
    this.notify("bar-padding");
  }

  get mipmap_bias(): number {
    return this._mipmap_bias;
  }

  set mipmap_bias(bias: number) {
    if (bias == this._mipmap_bias) return;
    this._mipmap_bias = bias;
    this.invalidate_contents();
    this.notify("mipmap-bias");
  }

  get peaks(): number[] | null {
    return this.peaks_mipmaps[0] || null;
  }

  set peaks(peaks: number[] | null) {
    if (peaks == null) {
      this.peaks_mipmaps = [];
    } else {
      this.peaks_mipmaps = [peaks];
    }
    this.invalidate_contents();
    this.notify("peaks");
  }

  vfunc_get_intrinsic_height(): number {
    return 0;
  }

  vfunc_get_intrinsic_width(): number {
    return (this.bar_width + 2 * this.bar_padding) * (this.peaks?.length || 0);
  }

  vfunc_get_intrinsic_aspect_ratio(): number {
    return 0;
  }

  vfunc_get_flags(): Gdk.PaintableFlags {
    return 0 as Gdk.PaintableFlags;
  }

  vfunc_get_current_image(): Gdk.Paintable {
    const snapshot = new Gtk.Snapshot();
    const width = this.vfunc_get_intrinsic_width();
    const height = 256;

    this.vfunc_snapshot(snapshot, width, height);

    return snapshot.to_paintable(
      new Graphene.Size({
        width: width,
        height: height,
      }),
    )!;
  }

  vfunc_snapshot(snapshot: Gtk.Snapshot, width: number, height: number): void {
    const { bar_padding, bar_width, color, peaks_mipmaps, mipmap_bias } = this;
    if (peaks_mipmaps.length <= 0) return;
    const peaks = peaks_mipmaps[0];

    const bar_total = bar_width + 2 * bar_padding;
    const peak_coefficient = height - 1;
    const rect = new Graphene.Rect();

    const needed_bars = Math.max(1, Math.floor(width / bar_total));
    const x_offset = Math.floor((width % bar_total) / 2 + bar_padding);

    const required_mipmap = Math.floor(
      Math.log2(peaks.length) - Math.log2(needed_bars) + mipmap_bias,
    );
    const used_mipmap = Math.max(0, required_mipmap);

    // Generate new mipmaps if needed
    while (peaks_mipmaps.length <= used_mipmap) {
      const current_mipmap =
        peaks_mipmaps.length == 0
          ? peaks
          : peaks_mipmaps[peaks_mipmaps.length - 1];
      const new_mipmap = new Array(
        Math.floor(current_mipmap.length / 2),
      ) as number[];

      for (let i = 0; i < new_mipmap.length; i += 1) {
        new_mipmap[i] = (current_mipmap[i * 2] + current_mipmap[i * 2 + 1]) / 2;
      }

      peaks_mipmaps.push(new_mipmap);
    }

    const mipmap = peaks_mipmaps[used_mipmap];
    const scaling_ratio = mipmap.length / needed_bars;

    let draw_x = x_offset;
    for (let i = 0; i <= needed_bars; i += 1) {
      // Linear interpolation between the two nearest peaks
      const peaks_x = i * scaling_ratio;
      const peaks_y1 = mipmap[Math.floor(peaks_x)] || 0;
      const peaks_y2 = mipmap[Math.ceil(peaks_x)] || 0;
      const lerp_factor = peaks_x % 1;
      const interpolated_value =
        peaks_y1 * (1 - lerp_factor) + peaks_y2 * lerp_factor;

      // Draw bar
      const bar_height = 1 + interpolated_value * peak_coefficient;
      rect.init(draw_x, (height - bar_height) / 2, bar_width, bar_height);
      snapshot.append_color(color, rect);
      draw_x += bar_total;
    }
  }

  // Interface Gdk.Paintable

  // @ts-expect-error Gdk.Paintable
  compute_concrete_size(
    specified_width: number,
    specified_height: number,
    default_width: number,
    default_height: number,
  ): [number, number];
  // @ts-expect-error Gdk.Paintable
  get_current_image(): Gdk.Paintable;
  // @ts-expect-error Gdk.Paintable
  get_flags(): Gdk.PaintableFlags;
  // @ts-expect-error Gdk.Paintable
  get_intrinsic_aspect_ratio(): number;
  // @ts-expect-error Gdk.Paintable
  get_intrinsic_height(): number;
  // @ts-expect-error Gdk.Paintable
  get_intrinsic_width(): number;
  // @ts-expect-error Gdk.Paintable
  invalidate_contents(): void;
  // @ts-expect-error Gdk.Paintable
  invalidate_size(): void;
  // @ts-expect-error Gdk.Paintable
  snapshot(snapshot: Gdk.Snapshot, width: number, height: number): void;
}
