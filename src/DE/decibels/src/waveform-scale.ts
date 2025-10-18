import Adw from "gi://Adw";
import GObject from "gi://GObject";
import Gdk from "gi://Gdk?version=4.0";
import Gsk from "gi://Gsk";
import Gtk from "gi://Gtk?version=4.0";
import Graphene from "gi://Graphene";

import { APWaveformBarChart } from "./waveform-bar-chart.js";

/**
 * A slider control similar to {@link Gtk.Scale} showing an audio waveform in the background.
 */
export class APWaveformScale extends Gtk.Widget {
  private _position: number = 0;
  private _paintable: APWaveformBarChart;
  private paintable_id: number;
  private notify_peaks_id: number;
  private _height_mul!: number;

  private left_color!: Gdk.RGBA;
  private right_color!: Gdk.RGBA;
  private cached_paintable?: Gdk.Paintable;
  private last_play_x: number = 0;

  private animation: Adw.TimedAnimation;

  static {
    GObject.registerClass(
      {
        GTypeName: "APWaveformScale",
        CssName: "waveform-scale",
        Properties: {
          position: GObject.ParamSpec.float(
            "position",
            "Waveform position",
            "Waveform position",
            GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
            0.0,
            1.0,
            0.0,
          ),
          paintable: GObject.param_spec_object(
            "paintable",
            "Paintable",
            "Internal child used for drawing the bar chart",
            APWaveformBarChart.$gtype,
            GObject.ParamFlags.READABLE,
          ),
          height_mul: GObject.ParamSpec.float(
            "height-mul",
            "height-mul",
            "Multiplier for the display height, used for animations",
            GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
            0.0,
            1.0,
            0.0,
          ),
        },
        Signals: {
          "position-changed": { param_types: [GObject.TYPE_DOUBLE] },
        },
      },
      this,
    );
  }

  constructor(params: Partial<Gtk.Widget.ConstructorProperties> | undefined) {
    super(params);

    this._paintable = new APWaveformBarChart();

    this.animation = Adw.TimedAnimation.new(
      this,
      0,
      1,
      250,
      Adw.PropertyAnimationTarget.new(this, "height-mul"),
    );
    this.animation.easing = Adw.Easing.EASE_IN_QUAD;

    this.paintable_id = this._paintable.connect("invalidate-contents", () => {
      this.cached_paintable = undefined;
      this.queue_draw();
    });

    this.notify_peaks_id = this._paintable.connect("notify::peaks", () => {
      this.animation.play();
    });

    const click_gesture = Gtk.GestureClick.new();
    click_gesture.connect("pressed", this.pressed_cb.bind(this));
    this.add_controller(click_gesture);

    const drag_gesture = Gtk.GestureDrag.new();
    drag_gesture.connect("drag-begin", this.drag_begin_cb.bind(this));
    drag_gesture.connect("drag-update", this.drag_update_cb.bind(this));
    this.add_controller(drag_gesture);

    this.update_colors();

    // @ts-expect-error: No overload for creating new GValue
    const gvalue_zero = new GObject.Value();
    gvalue_zero.init(GObject.TYPE_DOUBLE);
    gvalue_zero.set_double(0);
    this.update_property([Gtk.AccessibleProperty.VALUE_MIN], [gvalue_zero]);

    // @ts-expect-error: No overload for creating new GValue
    const gvalue_one = new GObject.Value();
    gvalue_one.init(GObject.TYPE_DOUBLE);
    gvalue_one.set_double(1);
    this.update_property([Gtk.AccessibleProperty.VALUE_MAX], [gvalue_one]);

    this.accessible_role = Gtk.AccessibleRole.SLIDER;
  }

  get paintable(): APWaveformBarChart {
    return this._paintable;
  }

  get position(): number {
    return this._position;
  }

  set position(position: number) {
    this._position = position;
    this.notify("position");

    if (Math.round(this.get_width() * position) !== this.last_play_x) {
      this.queue_draw();
    }

    // @ts-expect-error: No overload for creating new GValue
    const gvalue_position = new GObject.Value();
    gvalue_position.init(GObject.TYPE_DOUBLE);
    gvalue_position.set_double(position);
    this.update_property([Gtk.AccessibleProperty.VALUE_NOW], [gvalue_position]);
  }

  get height_mul(): number {
    return this._height_mul;
  }

  set height_mul(height_mul: number) {
    this._height_mul = height_mul;
    this.notify("height_mul");
    this.queue_draw();
  }

  private drag_begin_cb(gesture: Gtk.GestureDrag): void {
    gesture.set_state(Gtk.EventSequenceState.CLAIMED);
  }

  private drag_update_cb(gesture: Gtk.GestureDrag, offset_x: number): void {
    this.position = Math.max(
      0,
      Math.min(1, (gesture.get_start_point()[1] + offset_x) / this.get_width()),
    );
    this.emit("position-changed", this.position);
  }

  private pressed_cb(
    _gesture: Gtk.GestureClick,
    _n_press: number,
    x: number,
    _y: number,
  ): void {
    this.position = Math.max(0, Math.min(1, x / this.get_width()));
    this.emit("position-changed", this.position);
  }

  vfunc_css_changed(change: Gtk.CssStyleChange): void {
    super.vfunc_css_changed(change);
    this.update_colors();
  }

  private update_colors(): void {
    const style_manager = Adw.StyleManager.get_default();
    const { accent_color, dark } = style_manager;
    const current_color = this.get_color();
    current_color.alpha *= style_manager.high_contrast ? 0.9 : 0.55;

    this.left_color = Adw.accent_color_to_standalone_rgba(accent_color, dark);
    this.right_color = current_color;
    this.queue_draw();
  }

  vfunc_size_allocate(width: number, height: number, _baseline: number): void {
    if (
      height != this.cached_paintable?.get_intrinsic_height() ||
      width != this.cached_paintable?.get_intrinsic_width()
    ) {
      this.cached_paintable = undefined;
      this.queue_draw();
    }
  }

  vfunc_snapshot(snapshot: Gtk.Snapshot): void {
    if (this.paintable == undefined) return;

    const height = this.get_height();
    const width = this.get_width();
    const position = this.position;
    const height_mul = this.height_mul;
    if (height <= 0 || width <= 0) return;

    let cache = this.cached_paintable;
    if (cache == undefined) {
      const cache_snapshot = new Gtk.Snapshot();

      this.paintable.snapshot(cache_snapshot, width, height);
      cache = cache_snapshot.to_paintable(
        new Graphene.Size({
          width: width,
          height: height,
        }),
      )!;
      this.cached_paintable = cache;
    }
    const rect = new Graphene.Rect();
    const play_x = Math.round(width * position);
    this.last_play_x = play_x;

    // Draw monochrome bar chart as mask
    snapshot.push_mask(Gsk.MaskMode.ALPHA);
    if (height_mul === 1) {
      cache?.snapshot(snapshot, width, height);
    } else {
      snapshot.save();
      snapshot.translate(
        new Graphene.Point({ x: 0, y: (height * (1 - height_mul)) / 2 }),
      );
      snapshot.scale(1, height_mul);
      cache?.snapshot(snapshot, width, height);
      snapshot.restore();
    }

    snapshot.pop();

    // Paint bar chart with accent/current colors

    rect.init(0, 0, play_x, height);
    snapshot.append_color(this.left_color, rect);
    rect.init(play_x, 0, width - play_x, height);
    snapshot.append_color(this.right_color, rect);
    snapshot.pop();

    // Draw slider

    rect.init(play_x, 0, 2, height);
    snapshot.append_color(this.left_color, rect);
  }

  public destroy(): void {
    this.paintable.disconnect(this.paintable_id);
    this.paintable.disconnect(this.notify_peaks_id);
  }
}
