/*
 * mks-display-picture.c
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#define G_LOG_DOMAIN "mks-display-picture"

#include "config.h"

#include "mks-display-picture-private.h"
#include "mks-keyboard.h"
#include "mks-mouse.h"
#include "mks-touchable.h"
#include "mks-util-private.h"

struct _MksDisplayPicture
{
  GtkWidget     parent_instance;

  GSignalGroup *paintable_signals;
  MksPaintable *paintable;
  MksKeyboard  *keyboard;
  MksMouse     *mouse;
  MksTouchable *touchable;

  double        last_mouse_x;
  double        last_mouse_y;
};

enum {
  PROP_0,
  PROP_PAINTABLE,
  PROP_KEYBOARD,
  PROP_MOUSE,
  PROP_TOUCHABLE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (MksDisplayPicture, mks_display_picture, GTK_TYPE_WIDGET)

static GParamSpec *properties [N_PROPS];

static void
mks_display_picture_keyboard_press_cb (GObject      *object,
                                       GAsyncResult *result,
                                       gpointer      user_data)
{
  MksKeyboard *keyboard = (MksKeyboard *)object;
  g_autoptr(MksDisplayPicture) self = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (MKS_IS_KEYBOARD (keyboard));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (MKS_IS_DISPLAY_PICTURE (self));

  if (!mks_keyboard_press_finish (keyboard, result, &error))
    g_debug ("Keyboard press failed: %s", error->message);
}

static void
mks_display_picture_keyboard_release_cb (GObject      *object,
                                         GAsyncResult *result,
                                         gpointer      user_data)
{
  MksKeyboard *keyboard = (MksKeyboard *)object;
  g_autoptr(MksDisplayPicture) self = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (MKS_IS_KEYBOARD (keyboard));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (MKS_IS_DISPLAY_PICTURE (self));

  if (!mks_keyboard_release_finish (keyboard, result, &error))
    g_debug ("Keyboard release failed: %s", error->message);
}

static void
mks_display_picture_mouse_move_to_cb (GObject      *object,
                                      GAsyncResult *result,
                                      gpointer      user_data)
{
  MksMouse *mouse = (MksMouse *)object;
  g_autoptr(MksDisplayPicture) self = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (MKS_IS_MOUSE (mouse));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (MKS_IS_DISPLAY_PICTURE (self));

  if (!mks_mouse_move_to_finish (mouse, result, &error))
    g_debug ("Failed move_to: %s", error->message);
}

static void
mks_display_picture_mouse_move_by_cb (GObject      *object,
                                      GAsyncResult *result,
                                      gpointer      user_data)
{
  MksMouse *mouse = (MksMouse *)object;
  g_autoptr(MksDisplayPicture) self = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (MKS_IS_MOUSE (mouse));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (MKS_IS_DISPLAY_PICTURE (self));

  if (!mks_mouse_move_by_finish (mouse, result, &error))
    g_debug ("Failed move_by: %s", error->message);
}

static void
mks_display_picture_touchable_send_event_cb (GObject      *object,
                                             GAsyncResult *result,
                                             gpointer      user_data)
{
  MksTouchable *touchable = (MksTouchable *)object;
  g_autoptr(MksDisplayPicture) self = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (MKS_IS_TOUCHABLE (touchable));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (MKS_IS_DISPLAY_PICTURE (self));

  if (!mks_touchable_send_event_finish (touchable, result, &error))
    g_debug ("Failed to send touch event: %s", error->message);
}

static void
mks_display_picture_translate_button (MksDisplayPicture *self,
                                      int               *button)
{
  g_assert (MKS_IS_DISPLAY_PICTURE (self));
  g_assert (button != NULL);

  switch (*button)
    {
    case 1: *button = MKS_MOUSE_BUTTON_LEFT;   break;
    case 2: *button = MKS_MOUSE_BUTTON_MIDDLE; break;
    case 3: *button = MKS_MOUSE_BUTTON_RIGHT;  break;
    case 8: *button = MKS_MOUSE_BUTTON_SIDE;   break;
    case 9: *button = MKS_MOUSE_BUTTON_EXTRA;  break;
    default: break;
    }
}

static void
mks_display_picture_mouse_press_cb (GObject      *object,
                                    GAsyncResult *result,
                                    gpointer      user_data)
{
  MksMouse *mouse = (MksMouse *)object;
  g_autoptr(MksDisplayPicture) self = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (MKS_IS_MOUSE (mouse));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (MKS_IS_DISPLAY_PICTURE (self));

  if (!mks_mouse_press_finish (mouse, result, &error))
    g_debug ("Mouse press failed: %s", error->message);
}

static void
mks_display_picture_mouse_release_cb (GObject      *object,
                                      GAsyncResult *result,
                                      gpointer      user_data)
{
  MksMouse *mouse = (MksMouse *)object;
  g_autoptr(MksDisplayPicture) self = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (MKS_IS_MOUSE (mouse));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (MKS_IS_DISPLAY_PICTURE (self));

  if (!mks_mouse_release_finish (mouse, result, &error))
    g_debug ("Mouse release failed: %s", error->message);
}

gboolean
mks_display_picture_event_get_guest_position (MksDisplayPicture *self,
                                              GdkEvent          *event,
                                              double            *guest_x,
                                              double            *guest_y)
{
  GdkPaintable *paintable;
  GtkNative *native;
  int guest_width, guest_height;
  graphene_rect_t area;
  graphene_point_t translated;
  double translate_x, translate_y;
  double x, y;

  g_assert (MKS_IS_DISPLAY_PICTURE (self));
  g_assert (GDK_IS_EVENT (event));

  paintable = GDK_PAINTABLE (self->paintable);
  native = gtk_widget_get_native (GTK_WIDGET (self));
  guest_width = gdk_paintable_get_intrinsic_width (paintable);
  guest_height = gdk_paintable_get_intrinsic_height (paintable);

  area = GRAPHENE_RECT_INIT (0, 0,
                             gtk_widget_get_width (GTK_WIDGET (self)),
                             gtk_widget_get_height (GTK_WIDGET (self)));
  gtk_native_get_surface_transform (native, &translate_x, &translate_y);

  if (gdk_event_get_position (event, &x, &y))
    {
      x -= translate_x;
      y -= translate_y;

      if (!gtk_widget_compute_point (GTK_WIDGET (native),
                                     GTK_WIDGET (self),
                                     &GRAPHENE_POINT_INIT (x, y),
                                     &translated))
        return FALSE;

      *guest_x = floor (translated.x) / area.size.width * guest_width;
      *guest_y = floor (translated.y) / area.size.height * guest_height;

      *guest_x = CLAMP (*guest_x, 0, guest_width);
      *guest_y = CLAMP (*guest_y, 0, guest_width);

      return TRUE;
    }

  return FALSE;
}

static gboolean
mks_display_picture_legacy_event_cb (MksDisplayPicture        *self,
                                     GdkEvent                 *event,
                                     GtkEventControllerLegacy *controller)
{
  GdkPaintable *paintable;
  GdkEventType event_type;
  GdkEventSequence *sequence;

  g_assert (MKS_IS_DISPLAY_PICTURE (self));
  g_assert (GDK_IS_EVENT (event));
  g_assert (GTK_IS_EVENT_CONTROLLER_LEGACY (controller));

  if (self->keyboard == NULL || self->mouse == NULL || self->touchable == NULL || self->paintable == NULL)
    return GDK_EVENT_PROPAGATE;

  event_type = gdk_event_get_event_type (event);
  paintable = GDK_PAINTABLE (self->paintable);
  sequence = gdk_event_get_event_sequence (event);

  switch ((int)event_type)
    {
    case GDK_TOUCH_BEGIN:
    case GDK_TOUCH_UPDATE:
    case GDK_TOUCH_CANCEL:
    case GDK_TOUCH_END:
      {
        double guest_x, guest_y;
        guint64 num_slot = GPOINTER_TO_UINT (sequence);
        MksTouchEventKind kind;

        if (event_type == GDK_TOUCH_BEGIN)
          kind = MKS_TOUCH_EVENT_BEGIN;
        else if (event_type == GDK_TOUCH_UPDATE)
          kind = MKS_TOUCH_EVENT_UPDATE;
        else if (event_type == GDK_TOUCH_CANCEL)
          kind = MKS_TOUCH_EVENT_CANCEL;
        else
          kind = MKS_TOUCH_EVENT_END;

        if (mks_display_picture_event_get_guest_position (self, event, &guest_x, &guest_y))
          {
            mks_touchable_send_event (self->touchable, kind,
                                      num_slot,
                                      guest_x, guest_y,
                                      NULL,
                                      mks_display_picture_touchable_send_event_cb,
                                      g_object_ref (self));
            return GDK_EVENT_STOP;
          }

        break;
      }
    case GDK_MOTION_NOTIFY:
      {
        GdkSurface *surface = gdk_event_get_surface (event);
        GtkNative *native = gtk_widget_get_native (GTK_WIDGET (self));
        int guest_width = gdk_paintable_get_intrinsic_width (paintable);
        int guest_height = gdk_paintable_get_intrinsic_height (paintable);
        graphene_rect_t area;
        double translate_x;
        double translate_y;

        g_assert (MKS_IS_MOUSE (self->mouse));
        g_assert (GDK_IS_SURFACE (surface));

        area = GRAPHENE_RECT_INIT (0, 0,
                                   gtk_widget_get_width (GTK_WIDGET (self)),
                                   gtk_widget_get_height (GTK_WIDGET (self)));

        gtk_native_get_surface_transform (native, &translate_x, &translate_y);

        if (mks_mouse_get_is_absolute (self->mouse))
          {
            double guest_x, guest_y;
            if (mks_display_picture_event_get_guest_position (self, event, &guest_x, &guest_y))
              {
                mks_mouse_move_to (self->mouse,
                                   guest_x,
                                   guest_y,
                                   NULL,
                                   mks_display_picture_mouse_move_to_cb,
                                   g_object_ref (self));

                return GDK_EVENT_STOP;
              }
          }
        else
          {
            double x, y;

            if (gdk_event_get_axis (event, GDK_AXIS_X, &x) &&
                gdk_event_get_axis (event, GDK_AXIS_Y, &y))
              {
                double delta_x = self->last_mouse_x - (x / area.size.width) * guest_width;
                double delta_y = self->last_mouse_y - (y / area.size.height) * guest_height;

                mks_mouse_move_by (self->mouse,
                                   delta_x,
                                   delta_y,
                                   NULL,
                                   mks_display_picture_mouse_move_by_cb,
                                   g_object_ref (self));

                return GDK_EVENT_STOP;
              }
          }

        break;
      }

    case GDK_BUTTON_PRESS:
    case GDK_BUTTON_RELEASE:
      {
        int button = gdk_button_event_get_button (event);

        g_assert (MKS_IS_MOUSE (self->mouse));

        mks_display_picture_translate_button (self, &button);

        if (event_type == GDK_BUTTON_PRESS)
          mks_mouse_press (self->mouse,
                           button,
                           NULL,
                           mks_display_picture_mouse_press_cb,
                           g_object_ref (self));
        else
          mks_mouse_release (self->mouse,
                             button,
                             NULL,
                             mks_display_picture_mouse_release_cb,
                             g_object_ref (self));

        return GDK_EVENT_STOP;
      }

    case GDK_KEY_PRESS:
    case GDK_KEY_RELEASE:
      {
        guint keycode = gdk_key_event_get_keycode (event);
        guint keyval = gdk_key_event_get_keyval (event);
        guint qkeycode;

        g_assert (MKS_IS_KEYBOARD (self->keyboard));

        mks_keyboard_translate (keyval, keycode, &qkeycode);

        if (event_type == GDK_KEY_PRESS)
          mks_keyboard_press (self->keyboard,
                              qkeycode,
                              NULL,
                              mks_display_picture_keyboard_press_cb,
                              g_object_ref (self));
        else
          mks_keyboard_release (self->keyboard,
                                qkeycode,
                                NULL,
                                mks_display_picture_keyboard_release_cb,
                                g_object_ref (self));

        return GDK_EVENT_STOP;
      }

    case GDK_SCROLL:
      {
        GdkScrollDirection direction = gdk_scroll_event_get_direction (event);
        gboolean inverted = mks_scroll_event_is_inverted (event);
        int button = -1;

        g_assert (MKS_IS_MOUSE (self->mouse));

        switch (direction)
          {
          case GDK_SCROLL_UP:
            button = MKS_MOUSE_BUTTON_WHEEL_UP;
            break;

          case GDK_SCROLL_DOWN:
            button = MKS_MOUSE_BUTTON_WHEEL_DOWN;
            break;

          case GDK_SCROLL_SMOOTH:
            {
              double delta_x;
              double delta_y;

              /*
               * Currently there is no touchpad D-Bus interface to communicate
               * with QEMU. That is something we would very much want to have
               * in the future so that we can do this properly.
               *
               * For now, we just "emulate" scroll events by looking at direction
               * and sending that across as wheel events. It's enough to be useful
               * but far from what we would really want in the long run.
               */

              gdk_scroll_event_get_deltas (event, &delta_x, &delta_y);

              if (delta_y < 0)
                button = MKS_MOUSE_BUTTON_WHEEL_DOWN;
              else if (delta_y > 0)
                button = MKS_MOUSE_BUTTON_WHEEL_UP;

              break;
            }

          case GDK_SCROLL_LEFT:
          case GDK_SCROLL_RIGHT:
          default:
            break;
          }

        if (button != -1)
          {
            if (inverted)
              {
                if (button == MKS_MOUSE_BUTTON_WHEEL_UP)
                  button = MKS_MOUSE_BUTTON_WHEEL_DOWN;
                else if (button == MKS_MOUSE_BUTTON_WHEEL_DOWN)
                  button = MKS_MOUSE_BUTTON_WHEEL_UP;
              }

            mks_mouse_press (self->mouse,
                             button,
                             NULL,
                             mks_display_picture_mouse_press_cb,
                             g_object_ref (self));

            return GDK_EVENT_STOP;
          }

        break;
      }

    default:
      break;
    }

  return GDK_EVENT_PROPAGATE;
}

static void
mks_display_picture_invalidate_contents_cb (MksDisplayPicture *self,
                                            MksPaintable      *paintable)
{
  g_assert (MKS_IS_DISPLAY_PICTURE (self));
  g_assert (MKS_IS_PAINTABLE (paintable));

  gtk_widget_queue_draw (GTK_WIDGET (self));
}

static void
mks_display_picture_invalidate_size_cb (MksDisplayPicture *self,
                                        MksPaintable      *paintable)
{
  g_assert (MKS_IS_DISPLAY_PICTURE (self));
  g_assert (MKS_IS_PAINTABLE (paintable));

  gtk_widget_queue_resize (GTK_WIDGET (self));
}

static void
mks_display_picture_notify_cursor_cb (MksDisplayPicture *self,
                                      GParamSpec        *pspec,
                                      MksPaintable      *paintable)
{
  GdkCursor *cursor;

  g_assert (MKS_IS_DISPLAY_PICTURE (self));
  g_assert (MKS_IS_PAINTABLE (paintable));

  cursor = _mks_paintable_get_cursor (paintable);

  gtk_widget_set_cursor (GTK_WIDGET (self), cursor);
}

static void
mks_display_picture_mouse_set_cb (MksDisplayPicture *self,
                                  int                x,
                                  int                y,
                                  MksPaintable      *paintable)
{
  g_assert (MKS_IS_DISPLAY_PICTURE (self));
  g_assert (MKS_IS_PAINTABLE (paintable));

  self->last_mouse_x = x;
  self->last_mouse_y = y;
}

static void
mks_display_picture_measure (GtkWidget      *widget,
                             GtkOrientation  orientation,
                             int             for_size,
                             int            *minimum,
                             int            *natural,
                             int            *minimum_baseline,
                             int            *natural_baseline)
{
  MksDisplayPicture *self = (MksDisplayPicture *)widget;
  GdkPaintable *paintable;
  double nat_width, nat_height;
  int default_width;
  int default_height;

  g_assert (MKS_IS_DISPLAY_PICTURE (self));

  if (self->paintable == NULL || for_size == 0)
    {
      *minimum = 0;
      *natural = 0;
      return;
    }

  paintable = GDK_PAINTABLE (self->paintable);

  default_width = gdk_paintable_get_intrinsic_width (paintable);
  default_height = gdk_paintable_get_intrinsic_width (paintable);

  if (default_width <= 0)
    default_width = 640;

  if (default_height <= 0)
    default_height = 480;

  if (orientation == GTK_ORIENTATION_HORIZONTAL)
    {
      gdk_paintable_compute_concrete_size (paintable,
                                           0,
                                           for_size < 0 ? 0 : for_size,
                                           default_width, default_height,
                                           &nat_width, &nat_height);
      *minimum = 0;
      *natural = ceil (nat_width);
    }
  else
    {
      gdk_paintable_compute_concrete_size (paintable,
                                           for_size < 0 ? 0 : for_size,
                                           0,
                                           default_width, default_height,
                                           &nat_width, &nat_height);
      *minimum = 0;
      *natural = ceil (nat_height);
    }
}

static GtkSizeRequestMode
mks_display_picture_get_request_mode (GtkWidget *widget)
{
  return GTK_SIZE_REQUEST_HEIGHT_FOR_WIDTH;
}

static void
mks_display_picture_snapshot (GtkWidget   *widget,
                              GtkSnapshot *snapshot)
{
  MksDisplayPicture *self = (MksDisplayPicture *)widget;

  if (self->paintable == NULL)
    return;

  gdk_paintable_snapshot (GDK_PAINTABLE (self->paintable),
                          snapshot,
                          gtk_widget_get_width (GTK_WIDGET (self)),
                          gtk_widget_get_height (GTK_WIDGET (self)));
}

static void
mks_display_picture_dispose (GObject *object)
{
  MksDisplayPicture *self = (MksDisplayPicture *)object;

  g_clear_object (&self->paintable);
  g_clear_object (&self->keyboard);
  g_clear_object (&self->mouse);
  g_clear_object (&self->paintable_signals);
  g_clear_object (&self->touchable);

  G_OBJECT_CLASS (mks_display_picture_parent_class)->dispose (object);
}

static void
mks_display_picture_get_property (GObject    *object,
                                  guint       prop_id,
                                  GValue     *value,
                                  GParamSpec *pspec)
{
  MksDisplayPicture *self = MKS_DISPLAY_PICTURE (object);

  switch (prop_id)
    {
    case PROP_KEYBOARD:
      g_value_set_object (value, mks_display_picture_get_keyboard (self));
      break;

    case PROP_MOUSE:
      g_value_set_object (value, mks_display_picture_get_mouse (self));
      break;

    case PROP_TOUCHABLE:
      g_value_set_object (value, mks_display_picture_get_touchable (self));
      break;

    case PROP_PAINTABLE:
      g_value_set_object (value, mks_display_picture_get_paintable (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
mks_display_picture_set_property (GObject      *object,
                                  guint         prop_id,
                                  const GValue *value,
                                  GParamSpec   *pspec)
{
  MksDisplayPicture *self = MKS_DISPLAY_PICTURE (object);

  switch (prop_id)
    {
    case PROP_KEYBOARD:
      mks_display_picture_set_keyboard (self, g_value_get_object (value));
      break;

    case PROP_MOUSE:
      mks_display_picture_set_mouse (self, g_value_get_object (value));
      break;

    case PROP_TOUCHABLE:
      mks_display_picture_set_touchable (self, g_value_get_object (value));
      break;

    case PROP_PAINTABLE:
      mks_display_picture_set_paintable (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
mks_display_picture_class_init (MksDisplayPictureClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  object_class->dispose = mks_display_picture_dispose;
  object_class->get_property = mks_display_picture_get_property;
  object_class->set_property = mks_display_picture_set_property;

  widget_class->measure = mks_display_picture_measure;
  widget_class->get_request_mode = mks_display_picture_get_request_mode;
  widget_class->snapshot = mks_display_picture_snapshot;

  properties[PROP_KEYBOARD] =
    g_param_spec_object ("keyboard", NULL, NULL,
                         MKS_TYPE_KEYBOARD,
                         (G_PARAM_READWRITE | G_PARAM_EXPLICIT_NOTIFY | G_PARAM_STATIC_STRINGS));

  properties[PROP_MOUSE] =
    g_param_spec_object ("mouse", NULL, NULL,
                         MKS_TYPE_MOUSE,
                         (G_PARAM_READWRITE | G_PARAM_EXPLICIT_NOTIFY | G_PARAM_STATIC_STRINGS));

  properties[PROP_TOUCHABLE] =
    g_param_spec_object ("touchable", NULL, NULL,
                         MKS_TYPE_TOUCHABLE,
                         (G_PARAM_READWRITE | G_PARAM_EXPLICIT_NOTIFY | G_PARAM_STATIC_STRINGS));

  properties[PROP_PAINTABLE] =
    g_param_spec_object ("paintable", NULL, NULL,
                         MKS_TYPE_PAINTABLE,
                         (G_PARAM_READWRITE | G_PARAM_EXPLICIT_NOTIFY | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
mks_display_picture_init (MksDisplayPicture *self)
{
  GtkEventController *controller;
  g_autoptr(GdkCursor) gdk_cursor = gdk_cursor_new_from_name ("none", NULL);

  controller = gtk_event_controller_legacy_new ();
  g_signal_connect_object (controller,
                           "event",
                           G_CALLBACK (mks_display_picture_legacy_event_cb),
                           self,
                           G_CONNECT_SWAPPED);
  gtk_event_controller_set_propagation_phase (controller, GTK_PHASE_CAPTURE);
  gtk_widget_add_controller (GTK_WIDGET (self), controller);

  self->paintable_signals = g_signal_group_new (MKS_TYPE_PAINTABLE);
  g_signal_group_connect_object (self->paintable_signals,
                                 "invalidate-contents",
                                 G_CALLBACK (mks_display_picture_invalidate_contents_cb),
                                 self,
                                 G_CONNECT_SWAPPED);
  g_signal_group_connect_object (self->paintable_signals,
                                 "invalidate-size",
                                 G_CALLBACK (mks_display_picture_invalidate_size_cb),
                                 self,
                                 G_CONNECT_SWAPPED);
  g_signal_group_connect_object (self->paintable_signals,
                                 "notify::cursor",
                                 G_CALLBACK (mks_display_picture_notify_cursor_cb),
                                 self,
                                 G_CONNECT_SWAPPED);
  g_signal_group_connect_object (self->paintable_signals,
                                 "mouse-set",
                                 G_CALLBACK (mks_display_picture_mouse_set_cb),
                                 self,
                                 G_CONNECT_SWAPPED);

  gtk_widget_set_cursor (GTK_WIDGET (self), gdk_cursor);
  gtk_widget_set_focusable (GTK_WIDGET (self), TRUE);
}

GtkWidget *
mks_display_picture_new (void)
{
  return g_object_new (MKS_TYPE_DISPLAY_PICTURE, NULL);
}

MksPaintable *
mks_display_picture_get_paintable (MksDisplayPicture *self)
{
  g_return_val_if_fail (MKS_IS_DISPLAY_PICTURE (self), NULL);

  return self->paintable;
}

void
mks_display_picture_set_paintable (MksDisplayPicture *self,
                                   MksPaintable      *paintable)
{
  g_return_if_fail (MKS_IS_DISPLAY_PICTURE (self));
  g_return_if_fail (!paintable || MKS_IS_PAINTABLE (paintable));

  if (g_set_object (&self->paintable, paintable))
    {
      g_signal_group_set_target (self->paintable_signals, paintable);
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_PAINTABLE]);
      gtk_widget_queue_resize (GTK_WIDGET (self));
    }
}

MksMouse *
mks_display_picture_get_mouse (MksDisplayPicture *self)
{
  g_return_val_if_fail (MKS_IS_DISPLAY_PICTURE (self), NULL);

  return self->mouse;
}

void
mks_display_picture_set_mouse (MksDisplayPicture *self,
                               MksMouse          *mouse)
{
  g_return_if_fail (MKS_IS_DISPLAY_PICTURE (self));
  g_return_if_fail (!mouse || MKS_IS_MOUSE (mouse));

  if (g_set_object (&self->mouse, mouse))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_MOUSE]);
}

MksKeyboard *
mks_display_picture_get_keyboard (MksDisplayPicture *self)
{
  g_return_val_if_fail (MKS_IS_DISPLAY_PICTURE (self), NULL);

  return self->keyboard;
}

void
mks_display_picture_set_keyboard (MksDisplayPicture *self,
                                  MksKeyboard       *keyboard)
{
  g_return_if_fail (MKS_IS_DISPLAY_PICTURE (self));
  g_return_if_fail (!keyboard || MKS_IS_KEYBOARD (keyboard));

  if (g_set_object (&self->keyboard, keyboard))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_KEYBOARD]);
}

MksTouchable *
mks_display_picture_get_touchable (MksDisplayPicture *self)
{
  g_return_val_if_fail (MKS_IS_DISPLAY_PICTURE (self), NULL);

  return self->touchable;
}

void
mks_display_picture_set_touchable (MksDisplayPicture *self,
                                   MksTouchable      *touchable)
{
  g_return_if_fail (MKS_IS_DISPLAY_PICTURE (self));
  g_return_if_fail (!touchable || MKS_IS_TOUCHABLE (touchable));

  if (g_set_object (&self->touchable, touchable))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_TOUCHABLE]);
}
