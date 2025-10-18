/*
 * mks-paintable.c
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

#include "config.h"

#include <errno.h>
#include <sys/socket.h>

#include <glib/gstdio.h>
#include <gtk/gtk.h>
#include <pixman.h>

#include "mks-cairo-framebuffer-private.h"
#include "mks-dmabuf-paintable-private.h"
#include "mks-paintable-private.h"
#include "mks-qemu.h"
#include "mks-util-private.h"

#include "mks-marshal.h"

struct _MksPaintable
{
  GObject               parent_instance;
  MksQemuListener      *listener;
  GDBusConnection      *connection;
  GdkDisplay           *display;
  GdkPaintable         *child;
  GdkCursor            *cursor;
  MksDmabufScanoutData *scanout_data;
  int                   mouse_x;
  int                   mouse_y;
  guint                 y_inverted : 1;
};

enum {
  PROP_0,
  PROP_CURSOR,
  PROP_PAINTABLE,
  N_PROPS
};

enum {
  MOUSE_SET,
  N_SIGNALS
};

static GParamSpec *properties [N_PROPS];
static guint signals [N_SIGNALS];

static cairo_format_t
_pixman_format_to_cairo_format (guint pixman_format)
{
  switch (pixman_format)
    {
#if _CAIRO_CHECK_VERSION(1, 17, 2)
    case PIXMAN_rgba_float:
      return CAIRO_FORMAT_RGBA128F;
    case PIXMAN_rgb_float:
      return CAIRO_FORMAT_RGB96F;
#endif

    case PIXMAN_a8r8g8b8:
      return CAIRO_FORMAT_ARGB32;
    case PIXMAN_x2r10g10b10:
      return CAIRO_FORMAT_RGB30;
    case PIXMAN_x8r8g8b8:
      return CAIRO_FORMAT_RGB24;
    case PIXMAN_a8:
      return CAIRO_FORMAT_A8;
    case PIXMAN_a1:
      return CAIRO_FORMAT_A1;
    case PIXMAN_r5g6b5:
      return CAIRO_FORMAT_RGB16_565;
    default:
      return 0;
  }
}

static int
mks_paintable_get_intrinsic_height (GdkPaintable *paintable)
{
  GdkPaintable *child = MKS_PAINTABLE (paintable)->child;

  return child ? gdk_paintable_get_intrinsic_height (child) : 0;
}

static int
mks_paintable_get_intrinsic_width (GdkPaintable *paintable)
{
  GdkPaintable *child = MKS_PAINTABLE (paintable)->child;

  return child ? gdk_paintable_get_intrinsic_width (child) : 0;
}

static double
mks_paintable_get_intrinsic_aspect_ratio (GdkPaintable *paintable)
{
  GdkPaintable *child = MKS_PAINTABLE (paintable)->child;

  return child ? gdk_paintable_get_intrinsic_aspect_ratio (child) : .0;
}

static void
mks_paintable_snapshot (GdkPaintable *paintable,
                        GdkSnapshot  *snapshot,
                        double        width,
                        double        height)
{
  MksPaintable *self = MKS_PAINTABLE (paintable);

  if (self->child != NULL)
    {
      if (MKS_IS_DMABUF_PAINTABLE (self->child) && !self->y_inverted)
        {
          gtk_snapshot_save (snapshot);
          gtk_snapshot_translate (snapshot, &GRAPHENE_POINT_INIT (0, height));
          gtk_snapshot_scale (snapshot, 1, -1);
          gdk_paintable_snapshot (self->child, snapshot, width, height);
          gtk_snapshot_restore (snapshot);
        }
      else
        {
          gdk_paintable_snapshot (self->child, snapshot, width, height);
        }
    }
}

static void
paintable_iface_init (GdkPaintableInterface *iface)
{
  iface->get_intrinsic_height = mks_paintable_get_intrinsic_height;
  iface->get_intrinsic_width = mks_paintable_get_intrinsic_width;
  iface->get_intrinsic_aspect_ratio = mks_paintable_get_intrinsic_aspect_ratio;
  iface->snapshot = mks_paintable_snapshot;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (MksPaintable, mks_paintable, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (GDK_TYPE_PAINTABLE, paintable_iface_init))

static void
mks_paintable_dispose (GObject *object)
{
  MksPaintable *self = (MksPaintable *)object;

  g_clear_object (&self->connection);
  g_clear_object (&self->listener);
  g_clear_object (&self->child);
  g_clear_object (&self->cursor);
  g_clear_object (&self->display);

  G_OBJECT_CLASS (mks_paintable_parent_class)->dispose (object);
}

static void
mks_paintable_get_property (GObject    *object,
                            guint       prop_id,
                            GValue     *value,
                            GParamSpec *pspec)
{
  MksPaintable *self = MKS_PAINTABLE (object);

  switch (prop_id)
    {
    case PROP_CURSOR:
      g_value_set_object (value, _mks_paintable_get_cursor (self));
      break;

    case PROP_PAINTABLE:
      g_value_set_object (value, self->child);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
mks_paintable_class_init (MksPaintableClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = mks_paintable_dispose;
  object_class->get_property = mks_paintable_get_property;

  properties [PROP_CURSOR] =
    g_param_spec_object ("cursor", NULL, NULL,
                         GDK_TYPE_CURSOR,
                         (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  properties [PROP_PAINTABLE] =
    g_param_spec_object ("paintable", NULL, NULL,
                         GDK_TYPE_PAINTABLE,
                         (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  signals [MOUSE_SET] =
    g_signal_new ("mouse-set",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL, NULL,
                  _mks_marshal_VOID__INT_INT,
                  G_TYPE_NONE, 2, G_TYPE_INT, G_TYPE_INT);
  g_signal_set_va_marshaller (signals [MOUSE_SET],
                              G_TYPE_FROM_CLASS (klass),
                              _mks_marshal_VOID__INT_INTv);
}

static void
mks_paintable_init (MksPaintable *self)
{
}

static void
mks_paintable_invalidate_contents_cb (MksPaintable *self,
                                      GdkPaintable *paintable)
{
  g_assert (MKS_IS_PAINTABLE (self));
  g_assert (GDK_IS_PAINTABLE (paintable));

  gdk_paintable_invalidate_contents (GDK_PAINTABLE (self));
}

static void
mks_paintable_invalidate_size_cb (MksPaintable *self,
                                  GdkPaintable *paintable)
{
  g_assert (MKS_IS_PAINTABLE (self));
  g_assert (GDK_IS_PAINTABLE (paintable));

  gdk_paintable_invalidate_size (GDK_PAINTABLE (self));
}

static void
mks_paintable_set_child (MksPaintable *self,
                         GdkPaintable *child)
{
  gboolean size_changed;

  g_assert (MKS_IS_PAINTABLE (self));
  g_assert (!child || GDK_IS_PAINTABLE (child));

  if (self->child == child)
    return;

  size_changed = self->child == NULL ||
                 child == NULL ||
                 gdk_paintable_get_intrinsic_width (self->child) != gdk_paintable_get_intrinsic_width (child) ||
                 gdk_paintable_get_intrinsic_height (self->child) != gdk_paintable_get_intrinsic_height (child);

  if (self->child != NULL)
    {
      g_signal_handlers_disconnect_by_func (self->child,
                                            G_CALLBACK (mks_paintable_invalidate_size_cb),
                                            self);
      g_signal_handlers_disconnect_by_func (self->child,
                                            G_CALLBACK (mks_paintable_invalidate_contents_cb),
                                            self);
      g_clear_object (&self->child);
    }

  if (child != NULL)
    {
      self->child = g_object_ref (child);
      g_signal_connect_object (self->child,
                               "invalidate-size",
                               G_CALLBACK (mks_paintable_invalidate_size_cb),
                               self,
                               G_CONNECT_SWAPPED);
      g_signal_connect_object (self->child,
                               "invalidate-contents",
                               G_CALLBACK (mks_paintable_invalidate_contents_cb),
                               self,
                               G_CONNECT_SWAPPED);
    }

  gdk_paintable_invalidate_contents (GDK_PAINTABLE (self));

  if (size_changed)
    gdk_paintable_invalidate_size (GDK_PAINTABLE (self));

  g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_PAINTABLE]);
}

static gboolean
mks_paintable_listener_scanout_dmabuf (MksPaintable          *self,
                                       GDBusMethodInvocation *invocation,
                                       GUnixFDList           *unix_fd_list,
                                       GVariant              *dmabuf,
                                       guint                  width,
                                       guint                  height,
                                       guint                  stride,
                                       guint                  fourcc,
                                       guint64                modifier,
                                       gboolean               y0_top,
                                       MksQemuListener       *listener)
{
  g_autoptr(MksDmabufPaintable) child = NULL;
  g_autoptr(GError) error = NULL;
  int dmabuf_fd = -1;
  MksDmabufScanoutData *scanout_data;
  guint handle;

  g_assert (MKS_IS_PAINTABLE (self));
  g_assert (G_IS_DBUS_METHOD_INVOCATION (invocation));
  g_assert (MKS_QEMU_IS_LISTENER (listener));
  g_assert (g_variant_is_of_type (dmabuf, G_VARIANT_TYPE_HANDLE));

  handle = g_variant_get_handle (dmabuf);

  if (handle >= g_unix_fd_list_get_length (unix_fd_list))
    {
      g_dbus_method_invocation_return_error_literal (invocation,
                                                     G_IO_ERROR,
                                                     G_IO_ERROR_INVALID_ARGUMENT,
                                                     "Invalid handle to DMA-BUF");
      return TRUE;
    }

  if (!MKS_IS_DMABUF_PAINTABLE (self->child))
    {
      child = mks_dmabuf_paintable_new ();
      mks_paintable_set_child (self, GDK_PAINTABLE (child));
    }

  if (-1 == (dmabuf_fd = g_unix_fd_list_get (unix_fd_list, handle, &error)))
    {
      g_dbus_method_invocation_return_gerror (invocation, error);
      return TRUE;
    }

  self->y_inverted = !y0_top;
  scanout_data = g_new0 (MksDmabufScanoutData, 1);
  
  scanout_data->dmabuf_fd = dmabuf_fd;
  scanout_data->width = width;
  scanout_data->height = height;
  scanout_data->stride = stride;
  scanout_data->fourcc = fourcc;
  scanout_data->modifier = modifier;
  if (self->scanout_data)
    g_clear_fd (&self->scanout_data->dmabuf_fd, NULL);

  g_clear_pointer (&self->scanout_data, g_free);
  self->scanout_data = scanout_data;

  mks_qemu_listener_complete_scanout_dmabuf (listener, invocation, NULL);

  return TRUE;
}

static gboolean
mks_paintable_listener_update_dmabuf (MksPaintable          *self,
                                      GDBusMethodInvocation *invocation,
                                      int                    x,
                                      int                    y,
                                      int                    width,
                                      int                    height,
                                      MksQemuListener       *listener)
{
  cairo_region_t *region = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (MKS_IS_PAINTABLE (self));
  g_assert (G_IS_DBUS_METHOD_INVOCATION (invocation));
  g_assert (MKS_QEMU_IS_LISTENER (listener));

  if (MKS_IS_DMABUF_PAINTABLE (self->child))
    { 
      g_assert (self->scanout_data != NULL);
      if (!self->y_inverted)
        y = self->scanout_data->height - y - height;

      region = cairo_region_create_rectangle (&(cairo_rectangle_int_t) { x, y, width, height });
      if (!mks_dmabuf_paintable_import (MKS_DMABUF_PAINTABLE (self->child),
                                        self->display,
                                        self->scanout_data,
                                        region,
                                        &error))
        {
          g_dbus_method_invocation_return_gerror (invocation, error);
          goto cleanup;
        }
    }
  
  mks_qemu_listener_complete_update_dmabuf (listener, invocation);
cleanup:
  g_clear_pointer (&region, cairo_region_destroy);

  return TRUE;
}

static gboolean
mks_paintable_listener_update (MksPaintable          *self,
                               GDBusMethodInvocation *invocation,
                               int                    x,
                               int                    y,
                               int                    width,
                               int                    height,
                               guint                  stride,
                               guint                  pixman_format,
                               GVariant              *bytestring,
                               MksQemuListener       *listener)
{
  g_autoptr(GBytes) bytes = NULL;
  cairo_surface_t *source;
  const guint8 *data;
  cairo_t *cr;
  cairo_format_t format;
  gsize data_len;

  g_assert (MKS_IS_PAINTABLE (self));
  g_assert (G_IS_DBUS_METHOD_INVOCATION (invocation));
  g_assert (MKS_QEMU_IS_LISTENER (listener));

  if (!MKS_IS_CAIRO_FRAMEBUFFER (self->child) ||
      !(format = _pixman_format_to_cairo_format (pixman_format)))
    {
      g_dbus_method_invocation_return_error_literal (invocation,
                                                     G_IO_ERROR,
                                                     G_IO_ERROR_NOT_SUPPORTED,
                                                     "Invalid operation");
      return TRUE;
    }

  bytes = g_variant_get_data_as_bytes (bytestring);
  data = g_bytes_get_data (bytes, &data_len);

  if (data_len < cairo_format_stride_for_width (format, width) * height)
    {
      g_dbus_method_invocation_return_error_literal (invocation,
                                                     G_IO_ERROR,
                                                     G_IO_ERROR_NOT_SUPPORTED,
                                                     "Stride invalid for size");
      return TRUE;
    }

  /* We can get in a protocol race condition here in that we will get updates
   * for framebuffer content _BEFORE_ we'll get notified of property changes
   * about the MksQemuConsole's size.
   *
   * To overcome that, if we detect something larger than our current
   * framebuffer, we'll resize it and draw over the old contents in a
   * new framebuffer.
   *
   * When shrinking, we can do this as well and then handle it when the
   * console size notification arrives.
   *
   * Generally this is seen at startup during EFI/BIOS.
   */
  if (x + width > gdk_paintable_get_intrinsic_width (self->child) ||
      y + height > gdk_paintable_get_intrinsic_height (self->child))
    {
      guint max_width = MAX (gdk_paintable_get_intrinsic_width (self->child), x + width);
      guint max_height = MAX (gdk_paintable_get_intrinsic_height (self->child), y + height);
      g_autoptr(MksCairoFramebuffer) framebuffer = mks_cairo_framebuffer_new (format, max_width, max_height);

      mks_cairo_framebuffer_copy_to (MKS_CAIRO_FRAMEBUFFER (self->child), framebuffer);
      mks_paintable_set_child (self, GDK_PAINTABLE (framebuffer));
    }

  source = cairo_image_surface_create_for_data ((guint8 *)data, format, width, height, stride);
  cr = mks_cairo_framebuffer_update (MKS_CAIRO_FRAMEBUFFER (self->child), x, y, width, height);
  cairo_set_operator (cr, CAIRO_OPERATOR_SOURCE);
  cairo_set_source_surface (cr, source, 0, 0);
  cairo_rectangle (cr, 0, 0, width, height);
  cairo_paint (cr);
  cairo_destroy (cr);
  cairo_surface_destroy (source);

  mks_qemu_listener_complete_update (listener, invocation);

  return TRUE;
}

static gboolean
mks_paintable_listener_scanout (MksPaintable          *self,
                                GDBusMethodInvocation *invocation,
                                guint                  width,
                                guint                  height,
                                guint                  stride,
                                guint                  pixman_format,
                                GVariant              *bytestring,
                                MksQemuListener       *listener)
{
  g_autoptr(GBytes) bytes = NULL;
  cairo_surface_t *source;
  const guint8 *data;
  cairo_t *cr;
  cairo_format_t format;
  gsize data_len;

  g_assert (MKS_IS_PAINTABLE (self));
  g_assert (G_IS_DBUS_METHOD_INVOCATION (invocation));
  g_assert (MKS_QEMU_IS_LISTENER (listener));
  g_assert (g_variant_is_of_type (bytestring, G_VARIANT_TYPE_BYTESTRING));

  if (!(format = _pixman_format_to_cairo_format (pixman_format)))
    {
      g_dbus_method_invocation_return_error_literal (invocation,
                                                     G_IO_ERROR,
                                                     G_IO_ERROR_NOT_SUPPORTED,
                                                     "Pixman format not supported");
      return TRUE;
    }

  bytes = g_variant_get_data_as_bytes (bytestring);
  data = g_bytes_get_data (bytes, &data_len);

  if (data_len < cairo_format_stride_for_width (format, width) * height)
    {
      g_dbus_method_invocation_return_error_literal (invocation,
                                                     G_IO_ERROR,
                                                     G_IO_ERROR_NOT_SUPPORTED,
                                                     "Stride invalid for size");
      return TRUE;
    }

  if (self->child == NULL ||
      !MKS_IS_CAIRO_FRAMEBUFFER (self->child) ||
      width != gdk_paintable_get_intrinsic_width (self->child) ||
      height != gdk_paintable_get_intrinsic_height (self->child))
    {
      g_autoptr(MksCairoFramebuffer) child = mks_cairo_framebuffer_new (format, width, height);

      mks_paintable_set_child (self, GDK_PAINTABLE (child));
    }

  self->y_inverted = FALSE;

  source = cairo_image_surface_create_for_data ((guint8 *)data, format, width, height, stride);
  cr = mks_cairo_framebuffer_update (MKS_CAIRO_FRAMEBUFFER (self->child), 0, 0, width, height);
  cairo_set_source_surface (cr, source, 0, 0);
  cairo_set_operator (cr, CAIRO_OPERATOR_SOURCE);
  cairo_rectangle (cr, 0, 0, width, height);
  cairo_paint (cr);
  cairo_destroy (cr);
  cairo_surface_destroy (source);

  mks_qemu_listener_complete_scanout (listener, invocation);

  return TRUE;
}

static gboolean
mks_paintable_listener_cursor_define (MksPaintable          *self,
                                      GDBusMethodInvocation *invocation,
                                      int                    width,
                                      int                    height,
                                      int                    hot_x,
                                      int                    hot_y,
                                      GVariant              *bytestring,
                                      MksQemuListener       *listener)
{
  g_autoptr(GBytes) bytes = NULL;
  g_autoptr(GdkTexture) texture = NULL;
  g_autoptr(GdkCursor) cursor = NULL;
  gsize data_len;

  g_assert (MKS_IS_PAINTABLE (self));
  g_assert (G_IS_DBUS_METHOD_INVOCATION (invocation));
  g_assert (MKS_QEMU_IS_LISTENER (listener));

  if (width < 1 || width > 512 ||
      height < 1 || height > 512 ||
      !(bytes = g_variant_get_data_as_bytes (bytestring)))
    goto failure;

  data_len = g_bytes_get_size (bytes);
  if (data_len != (4 * width * height))
    goto failure;

  texture = gdk_memory_texture_new (width,
                                    height,
#if G_BYTE_ORDER == G_LITTLE_ENDIAN
                                    GDK_MEMORY_B8G8R8A8_PREMULTIPLIED,
#else
                                    GDK_MEMORY_A8R8G8B8_PREMULTIPLIED,
#endif
                                    bytes,
                                    width * 4);

  cursor = gdk_cursor_new_from_texture (texture, hot_x, hot_y, NULL);

  if (g_set_object (&self->cursor, cursor))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_CURSOR]);

failure:
  mks_qemu_listener_complete_cursor_define (listener, invocation);

  return TRUE;
}

static gboolean
mks_paintable_listener_mouse_set (MksPaintable          *self,
                                  GDBusMethodInvocation *invocation,
                                  int                    x,
                                  int                    y,
                                  int                    on,
                                  MksQemuListener       *listener)
{
  g_assert (MKS_IS_PAINTABLE (self));
  g_assert (G_IS_DBUS_METHOD_INVOCATION (invocation));
  g_assert (MKS_QEMU_IS_LISTENER (listener));

  self->mouse_x = x;
  self->mouse_y = y;

  mks_qemu_listener_complete_mouse_set (listener, invocation);

  g_signal_emit (self, signals[MOUSE_SET], 0, x, y);

  return TRUE;
}

static gboolean
mks_paintable_listener_disable (MksPaintable          *self,
                                GDBusMethodInvocation *invocation,
                                MksQemuListener       *listener)
{
  g_assert (MKS_IS_PAINTABLE (self));
  g_assert (G_IS_DBUS_METHOD_INVOCATION (invocation));
  g_assert (MKS_QEMU_IS_LISTENER (listener));

  if (MKS_IS_CAIRO_FRAMEBUFFER (self->child))
    mks_cairo_framebuffer_clear (MKS_CAIRO_FRAMEBUFFER (self->child));

  gdk_paintable_invalidate_contents (GDK_PAINTABLE (self));

  mks_qemu_listener_complete_disable (listener, invocation);

  return TRUE;
}


static void
mks_paintable_connection_cb (GObject      *object,
                             GAsyncResult *result,
                             gpointer      user_data)
{
  g_autoptr(MksPaintable) self = user_data;
  g_autoptr(GDBusConnection) connection = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (MKS_IS_PAINTABLE (self));
  g_assert (G_IS_ASYNC_RESULT (result));

  if (!(connection = g_dbus_connection_new_finish (result, &error)))
    {
      g_warning ("Failed to create D-Bus connection: %s", error->message);
      return;
    }

  g_set_object (&self->connection, connection);

  if (!g_dbus_interface_skeleton_export (G_DBUS_INTERFACE_SKELETON (self->listener),
                                         connection,
                                         "/org/qemu/Display1/Listener",
                                         &error))
    {
      g_warning ("Failed to export listener on bus: %s", error->message);
      return;
    }

  g_dbus_connection_start_message_processing (connection);

}

GdkPaintable *
_mks_paintable_new (GdkDisplay    *display,
                    GCancellable  *cancellable,
                    int           *peer_fd,
                    GError       **error)
{
  g_autoptr(MksPaintable) self = NULL;
  g_autoptr(GSocketConnection) io_stream = NULL;
  g_autoptr(GSocket) socket = NULL;
  g_autofd int us = -1;
  g_autofd int them = -1;

  g_return_val_if_fail (!cancellable || G_IS_CANCELLABLE (cancellable), NULL);
  g_return_val_if_fail (peer_fd != NULL, NULL);

  *peer_fd = -1;

  self = g_object_new (MKS_TYPE_PAINTABLE, NULL);
  self->display = g_object_ref (display);
  /* Create a socketpair() to use for D-Bus P2P protocol. We will be receiving
   * DMA-BUF FDs over this.
   */
  if (!mks_socketpair_create (&us, &them, error))
    return NULL;

  /* Create socket for our side of the socket pair */
  if (!(socket = g_socket_new_from_fd (us, error)))
    return NULL;
  us = -1;

  /* And convert that socket into a GIOStream */
  io_stream = g_socket_connection_factory_create_connection (socket);

  /* Setup our listener and callbacks to process requests */
  self->listener = mks_qemu_listener_skeleton_new ();
  g_signal_connect_object (self->listener,
                           "handle-scanout",
                           G_CALLBACK (mks_paintable_listener_scanout),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (self->listener,
                           "handle-update",
                           G_CALLBACK (mks_paintable_listener_update),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (self->listener,
                           "handle-scanout-dmabuf",
                           G_CALLBACK (mks_paintable_listener_scanout_dmabuf),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (self->listener,
                           "handle-update-dmabuf",
                           G_CALLBACK (mks_paintable_listener_update_dmabuf),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (self->listener,
                           "handle-disable",
                           G_CALLBACK (mks_paintable_listener_disable),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (self->listener,
                           "handle-cursor-define",
                           G_CALLBACK (mks_paintable_listener_cursor_define),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (self->listener,
                           "handle-mouse-set",
                           G_CALLBACK (mks_paintable_listener_mouse_set),
                           self,
                           G_CONNECT_SWAPPED);

  /* Asynchronously create connection because we can't do it synchronously
   * as the other side is doing AUTHENTICATION_SERVER for no good reason.
   */
  g_dbus_connection_new (G_IO_STREAM (io_stream),
                         NULL,
                         G_DBUS_CONNECTION_FLAGS_DELAY_MESSAGE_PROCESSING|G_DBUS_CONNECTION_FLAGS_AUTHENTICATION_CLIENT,
                         NULL,
                         cancellable,
                         mks_paintable_connection_cb,
                         g_object_ref (self));

  *peer_fd = g_steal_fd (&them);

  g_assert (*peer_fd != -1);
  g_assert (MKS_IS_PAINTABLE (self));
  g_assert (MKS_QEMU_IS_LISTENER (self->listener));

  return GDK_PAINTABLE (g_steal_pointer (&self));
}

/**
 * _mks_paintable_get_cursor:
 * @self: a #MksPaintable
 *
 * Gets the cursor as defined by the QEMU instance.
 *
 * Returns: (transfer none) (nullable): a #GdkCursor or %NULL
 */
GdkCursor *
_mks_paintable_get_cursor (MksPaintable *self)
{
  g_return_val_if_fail (MKS_IS_PAINTABLE (self), NULL);

  return self->cursor;
}
