/*
 * mks-screen.c
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

#include "mks-device-private.h"
#include "mks-enums.h"
#include "mks-qemu.h"
#include "mks-keyboard.h"
#include "mks-mouse.h"
#include "mks-paintable-private.h"
#include "mks-screen-attributes-private.h"
#include "mks-screen.h"
#include "mks-touchable.h"

struct _MksScreenClass
{
  MksDeviceClass parent_class;
};

struct _MksScreen
{
  MksDevice       parent_instance;

  MksQemuConsole *console;
  gulong          console_notify_handler;

  MksKeyboard    *keyboard;
  MksMouse       *mouse;
  MksTouchable   *touchable;

  guint           number;
  guint           width;
  guint           height;

  MksScreenKind   kind : 2;
};

G_DEFINE_FINAL_TYPE (MksScreen, mks_screen, MKS_TYPE_DEVICE)

enum {
  PROP_0,
  PROP_DEVICE_ADDRESS,
  PROP_HEIGHT,
  PROP_KIND,
  PROP_KEYBOARD,
  PROP_MOUSE,
  PROP_NUMBER,
  PROP_WIDTH,
  N_PROPS
};

static GParamSpec *properties [N_PROPS];

static void
mks_screen_set_width (MksScreen *self,
                      guint      width)
{
  g_assert (MKS_IS_SCREEN (self));

  if (self->width != width)
    {
      self->width = width;
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_WIDTH]);
    }
}

static void
mks_screen_set_height (MksScreen *self,
                       guint      height)
{
  g_assert (MKS_IS_SCREEN (self));

  if (self->height != height)
    {
      self->height = height;
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_HEIGHT]);
    }
}

static void
mks_screen_set_number (MksScreen *self,
                       guint      number)
{
  g_assert (MKS_IS_SCREEN (self));

  if (self->number != number)
    {
      self->number = number;
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_NUMBER]);
    }
}

static void
mks_screen_set_type (MksScreen  *self,
                     const char *type)
{
  MksScreenKind kind;

  g_assert (MKS_IS_SCREEN (self));

  kind = MKS_SCREEN_KIND_TEXT;

  if (strcmp (type, "Graphic") == 0)
    kind = MKS_SCREEN_KIND_GRAPHIC;

  if (kind != self->kind)
    {
      self->kind = kind;
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_KIND]);
    }
}

static void
mks_screen_console_notify_cb (MksScreen      *self,
                              GParamSpec     *pspec,
                              MksQemuConsole *console)
{
  g_assert (MKS_IS_SCREEN (self));
  g_assert (pspec != NULL);
  g_assert (MKS_QEMU_IS_CONSOLE (console));

  if (strcmp (pspec->name, "label") == 0)
    _mks_device_set_name (MKS_DEVICE (self), mks_qemu_console_get_label (console));
  else if (strcmp (pspec->name, "width") == 0)
    mks_screen_set_width (self, mks_qemu_console_get_width (console));
  else if (strcmp (pspec->name, "height") == 0)
    mks_screen_set_height (self, mks_qemu_console_get_height (console));
  else if (strcmp (pspec->name, "number") == 0)
    mks_screen_set_number (self, mks_qemu_console_get_head (console));
  else if (strcmp (pspec->name, "type") == 0)
    mks_screen_set_type (self, mks_qemu_console_get_type_ ((console)));
}

static void
mks_screen_set_console (MksScreen      *self,
                        MksQemuConsole *console)
{
  g_assert (MKS_IS_SCREEN (self));
  g_assert (!console || MKS_QEMU_IS_CONSOLE (console));

  if (self->console != NULL)
    return;

  if (g_set_object (&self->console, console))
    {
      _mks_device_set_name (MKS_DEVICE (self), mks_qemu_console_get_label (console));

      self->console_notify_handler =
        g_signal_connect_object (console,
                                 "notify",
                                 G_CALLBACK (mks_screen_console_notify_cb),
                                 self,
                                 G_CONNECT_SWAPPED);

      mks_screen_set_type (self, mks_qemu_console_get_type_ ((console)));
      mks_screen_set_width (self, mks_qemu_console_get_width (console));
      mks_screen_set_height (self, mks_qemu_console_get_height (console));
      mks_screen_set_number (self, mks_qemu_console_get_head (console));
    }
}

static gboolean
mks_screen_setup (MksDevice     *device,
                  MksQemuObject *object)
{
  MksScreen *self = (MksScreen *)device;
  g_autolist(GDBusInterface) interfaces = NULL;

  g_assert (MKS_IS_SCREEN (self));
  g_assert (MKS_QEMU_IS_OBJECT (object));

  interfaces = g_dbus_object_get_interfaces (G_DBUS_OBJECT (object));

  for (const GList *iter = interfaces; iter; iter = iter->next)
    {
      GDBusInterface *iface = iter->data;

      if (MKS_QEMU_IS_CONSOLE (iface))
        mks_screen_set_console (self, MKS_QEMU_CONSOLE (iface));
      else if (MKS_QEMU_IS_KEYBOARD (iface))
        self->keyboard = _mks_device_new (MKS_TYPE_KEYBOARD, device->session, object);
      else if (MKS_QEMU_IS_MOUSE (iface))
        self->mouse = _mks_device_new (MKS_TYPE_MOUSE, device->session, object);
      else if (MKS_QEMU_IS_MULTI_TOUCH (iface))
        self->touchable = _mks_device_new (MKS_TYPE_TOUCHABLE, device->session, object);
    }

  return self->console != NULL &&
         self->keyboard != NULL &&
         self->mouse != NULL;
}

static void
mks_screen_dispose (GObject *object)
{
  MksScreen *self = (MksScreen *)object;

  if (self->console != NULL)
    {
      g_clear_signal_handler (&self->console_notify_handler, self->console);
      g_clear_object (&self->console);
    }

  g_clear_object (&self->keyboard);
  g_clear_object (&self->mouse);
  g_clear_object (&self->touchable);

  G_OBJECT_CLASS (mks_screen_parent_class)->dispose (object);
}

static void
mks_screen_get_property (GObject    *object,
                         guint       prop_id,
                         GValue     *value,
                         GParamSpec *pspec)
{
  MksScreen *self = MKS_SCREEN (object);

  switch (prop_id)
    {
    case PROP_DEVICE_ADDRESS:
      g_value_set_string (value, mks_screen_get_device_address (self));
      break;

    case PROP_KEYBOARD:
      g_value_set_object (value, mks_screen_get_keyboard (self));
      break;

    case PROP_KIND:
      g_value_set_enum (value, mks_screen_get_kind (self));
      break;

    case PROP_MOUSE:
      g_value_set_object (value, mks_screen_get_mouse (self));
      break;

    case PROP_NUMBER:
      g_value_set_uint (value, mks_screen_get_number (self));
      break;

    case PROP_WIDTH:
      g_value_set_uint (value, mks_screen_get_width (self));
      break;

    case PROP_HEIGHT:
      g_value_set_uint (value, mks_screen_get_height (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
mks_screen_class_init (MksScreenClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  MksDeviceClass *device_class = MKS_DEVICE_CLASS (klass);

  object_class->dispose = mks_screen_dispose;
  object_class->get_property = mks_screen_get_property;

  device_class->setup = mks_screen_setup;

  properties [PROP_DEVICE_ADDRESS] =
    g_param_spec_string ("device-address", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  properties [PROP_KEYBOARD] =
    g_param_spec_object ("keyboard", NULL, NULL,
                         MKS_TYPE_KEYBOARD,
                         (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  properties [PROP_KIND] =
    g_param_spec_enum ("kind", NULL, NULL,
                       MKS_TYPE_SCREEN_KIND,
                       MKS_SCREEN_KIND_TEXT,
                       (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  properties [PROP_MOUSE] =
    g_param_spec_object ("mouse", NULL, NULL,
                         MKS_TYPE_MOUSE,
                         (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  properties [PROP_NUMBER] =
    g_param_spec_uint ("number", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  properties [PROP_WIDTH] =
    g_param_spec_uint ("width", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  properties [PROP_HEIGHT] =
    g_param_spec_uint ("height", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
mks_screen_init (MksScreen *self)
{
}

/**
 * mks_screen_get_keyboard:
 * @self: a #MksScreen
 *
 * Gets the #MksScreen:keyboard property.
 *
 * Returns: (transfer none): a #MksKeyboard
 */
MksKeyboard *
mks_screen_get_keyboard (MksScreen *self)
{
  g_return_val_if_fail (MKS_IS_SCREEN (self), NULL);

  return self->keyboard;
}

/**
 * mks_screen_get_mouse:
 * @self: a #MksScreen
 *
 * Gets the #MksScreen:mouse property.
 *
 * Returns: (transfer none): a #MksMouse
 */
MksMouse *
mks_screen_get_mouse (MksScreen *self)
{
  g_return_val_if_fail (MKS_IS_SCREEN (self), NULL);

  return self->mouse;
}

/**
 * mks_screen_get_touchable:
 * @self: a #MksScreen
 *
 * Gets the #MksScreen:touchable property.
 *
 * Returns: (transfer none): a #MksTouchable
 */
MksTouchable *
mks_screen_get_touchable (MksScreen *self)
{
  g_return_val_if_fail (MKS_IS_SCREEN (self), NULL);

  return self->touchable;
}

/**
 * mks_screen_get_kind:
 * @self: a #MksScreen
 *
 * Gets the "kind" property.
 *
 * Returns: a #MksScreenKind
 */
MksScreenKind
mks_screen_get_kind (MksScreen *self)
{
  g_return_val_if_fail (MKS_IS_SCREEN (self), MKS_SCREEN_KIND_TEXT);

  return self->kind;
}

/**
 * mks_screen_get_width:
 * @self: a #MksScreen
 *
 * Gets the "width" property.
 *
 * Returns: The width of the screen in pixels.
 */
guint
mks_screen_get_width (MksScreen *self)
{
  g_return_val_if_fail (MKS_IS_SCREEN (self), 0);

  return self->width;
}

/**
 * mks_screen_get_height:
 * @self: a #MksScreen
 *
 * Gets the "height" property.
 *
 * Returns: The height of the screen in pixels.
 */
guint
mks_screen_get_height (MksScreen *self)
{
  g_return_val_if_fail (MKS_IS_SCREEN (self), 0);

  return self->height;
}

/**
 * mks_screen_get_number:
 * @self: a #MksScreen
 *
 * Gets the "number" property.
 *
 * Returns: the screen number
 */
guint
mks_screen_get_number (MksScreen *self)
{
  g_return_val_if_fail (MKS_IS_SCREEN (self), 0);

  return self->number;
}

const char *
mks_screen_get_device_address (MksScreen *self)
{
  g_return_val_if_fail (MKS_IS_SCREEN (self), NULL);

  if (self->console != NULL)
    return mks_qemu_console_get_device_address (self->console);

  return NULL;
}

static gboolean
check_console (MksScreen  *self,
               GError    **error)
{
  if (self->console == NULL)
    {
      g_set_error_literal (error,
                           G_IO_ERROR,
                           G_IO_ERROR_NOT_CONNECTED,
                           "Not connected");
      return FALSE;
    }

  return TRUE;
}

static void
mks_screen_configure_cb (GObject      *object,
                         GAsyncResult *result,
                         gpointer      user_data)
{
  MksQemuConsole *console = (MksQemuConsole *)object;
  g_autoptr(GError) error = NULL;
  g_autoptr(GTask) task = user_data;

  g_assert (MKS_QEMU_IS_CONSOLE (console));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (G_IS_TASK (task));

  if (!mks_qemu_console_call_set_uiinfo_finish (console, result, &error))
    g_task_return_error (task, g_steal_pointer (&error));
  else
    g_task_return_boolean (task, TRUE);
}

/**
 * mks_screen_configure_async:
 * @self: an #MksScreen
 * @attributes: (transfer full): a #MksScreenAttributes
 * @cancellable: (nullable): a #GCancellable
 * @callback: a #GAsyncReadyCallback to execute upon completion
 * @user_data: closure data for @callback
 *
 * Requests the QEMU instance reconfigure the screen with @attributes.
 *
 * This function takes ownership of @attributes.
 *
 * @callback is executed upon acknowledgment from the QEMU instance or
 * if the request timed out.
 *
 * Call mks_screen_configure_finish() to get the result.
 */
void
mks_screen_configure (MksScreen           *self,
                      MksScreenAttributes *attributes,
                      GCancellable        *cancellable,
                      GAsyncReadyCallback  callback,
                      gpointer             user_data)
{
  g_autoptr(GTask) task = NULL;
  g_autoptr(GError) error = NULL;

  g_return_if_fail (MKS_IS_SCREEN (self));
  g_return_if_fail (attributes != NULL);
  g_return_if_fail (!cancellable || G_IS_CANCELLABLE (cancellable));

  task = g_task_new (self, cancellable, callback, user_data);
  g_task_set_source_tag (task, mks_screen_configure);

  if (!check_console (self, &error))
    g_task_return_error (task, g_steal_pointer (&error));
  else
    mks_qemu_console_call_set_uiinfo (self->console,
                                      attributes->width_mm,
                                      attributes->height_mm,
                                      attributes->x_offset,
                                      attributes->y_offset,
                                      attributes->width,
                                      attributes->height,
                                      cancellable,
                                      mks_screen_configure_cb,
                                      g_steal_pointer (&task));

  mks_screen_attributes_free (attributes);
}

/**
 * mks_screen_configure_finish:
 * @self: an #MksScreen
 * @result: a #GAsyncResult provided to callback
 * @error: a location for a #GError, or %NULL
 *
 * Completes a call to mks_screen_configure().
 *
 * Returns: %TRUE if the operation completed successfully; otherwise %FALSE
 *   and @error is set.
 */
gboolean
mks_screen_configure_finish (MksScreen     *self,
                             GAsyncResult  *result,
                             GError       **error)
{
  g_return_val_if_fail (MKS_IS_SCREEN (self), FALSE);
  g_return_val_if_fail (g_task_is_valid (result, self), FALSE);

  return g_task_propagate_boolean (G_TASK (result), error);
}

/**
 * mks_screen_configure_sync:
 * @self: a #MksScreen
 * @attributes: (transfer full): a #MksScreenAttributes
 * @cancellable: a #GCancellable
 * @error: a location for a #GError, or %NULL
 *
 * Requests the QEMU instance reconfigure the screen using @attributes.
 *
 * This function takes ownership of @attributes.
 *
 * Returns: %TRUE if the operation completed successfully; otherwise %FALSE
 *   and @error is set.
 */
gboolean
mks_screen_configure_sync (MksScreen            *self,
                           MksScreenAttributes  *attributes,
                           GCancellable         *cancellable,
                           GError              **error)
{
  g_return_val_if_fail (MKS_IS_SCREEN (self), FALSE);
  g_return_val_if_fail (attributes != NULL, FALSE);
  g_return_val_if_fail (!cancellable || G_IS_CANCELLABLE (cancellable), FALSE);

  if (!check_console (self, error))
    return FALSE;

  return mks_qemu_console_call_set_uiinfo_sync (self->console,
                                                attributes->width_mm,
                                                attributes->height_mm,
                                                attributes->x_offset,
                                                attributes->y_offset,
                                                attributes->width,
                                                attributes->height,
                                                cancellable,
                                                error);
}

static void
mks_screen_attach_cb (GObject      *object,
                      GAsyncResult *result,
                      gpointer      user_data)
{
  MksQemuConsole *console = (MksQemuConsole *)object;
  g_autoptr(GTask) task = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (G_IS_OBJECT (object));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (G_IS_TASK (task));

  if (!mks_qemu_console_call_register_listener_finish (console, NULL, result, &error))
    g_task_return_error (task, g_steal_pointer (&error));
  else
    g_task_return_pointer (task,
                           g_object_ref (g_task_get_task_data (task)),
                           g_object_unref);
}

/**
 * mks_screen_attach:
 * @self: an #MksScreen
 * @cancellable: (nullable): a #GCancellable
 * @callback: a #GAsyncReadyCallback to execute upon completion
 * @user_data: closure data for @callback
 *
 * Asynchronously creates a #GdkPaintable that is updated with the
 * contents of the screen.
 *
 * This function registers a new `socketpair()` which is shared with
 * the QEMU instance to receive rendering updates. Those updates are
 * propagated to the resulting #GdkPainable which can be retrieved
 * using mks_screen_attach_finish() from @callback.
 */
void
mks_screen_attach (MksScreen           *self,
                   GdkDisplay          *display,
                   GCancellable        *cancellable,
                   GAsyncReadyCallback  callback,
                   gpointer             user_data)
{
  g_autoptr(GUnixFDList) unix_fd_list = NULL;
  g_autoptr(GdkPaintable) paintable = NULL;
  g_autoptr(GTask) task = NULL;
  g_autoptr(GError) error = NULL;
  g_autofd int fd = -1;

  g_return_if_fail (MKS_IS_SCREEN (self));
  g_return_if_fail (!cancellable || G_IS_CANCELLABLE (cancellable));

  task = g_task_new (self, cancellable, callback, user_data);
  g_task_set_source_tag (task, mks_screen_attach);

  if (!check_console (self, &error) ||
      !(paintable = _mks_paintable_new (display, cancellable, &fd, &error)))
    goto failure;

  g_task_set_task_data (task, g_steal_pointer (&paintable), g_object_unref);

  unix_fd_list = g_unix_fd_list_new_from_array (&fd, 1), fd = -1;
  mks_qemu_console_call_register_listener (self->console,
                                           g_variant_new_handle (0),
                                           unix_fd_list,
                                           cancellable,
                                           mks_screen_attach_cb,
                                           g_steal_pointer (&task));

  return;

failure:
  g_task_return_error (task, g_steal_pointer (&error));
}

/**
 * mks_screen_attach_finish:
 * @self: an #MksScreen
 * @result: a #GAsyncResult provided to callback
 * @error: a location for a #GError, or %NULL
 *
 * Completes an asynchronous request to create a [iface@Gdk.Paintable] containing
 * the contents of #MksScreen in the QEMU instance.
 *
 * The resulting [iface@Gdk.Paintable] will be updated as changes are delivered
 * from QEMU over a private `socketpair()`. In the typical case, those
 * changes are propagated using a DMA-BUF and damage notifications.
 *
 * Returns: (transfer full): a #GdkPainable if successful; otherwise %NULL
 *   and @error is set.
 */
GdkPaintable *
mks_screen_attach_finish (MksScreen     *self,
                          GAsyncResult  *result,
                          GError       **error)
{
  g_return_val_if_fail (MKS_IS_SCREEN (self), FALSE);
  g_return_val_if_fail (g_task_is_valid (result, self), FALSE);

  return g_task_propagate_pointer (G_TASK (result), error);
}

/**
 * mks_screen_attach_sync:
 * @self: a #MksScreen
 * @cancellable: (nullable): a #GCancellable or %NULL
 * @error: (nullable): a location for a #GError, or %NULL
 *
 * Synchronous request to attach to screen, creating a paintable that can
 * be used to update display as the QEMU instance updates.
 *
 * Returns: (transfer full): a #GdkPaintable if successful; otherwise %NULL
 *   and @error is set.
 */
GdkPaintable *
mks_screen_attach_sync (MksScreen     *self,
                        GdkDisplay    *display,
                        GCancellable  *cancellable,
                        GError       **error)
{
  g_autoptr(GUnixFDList) unix_fd_list = NULL;
  g_autoptr(GdkPaintable) paintable = NULL;
  g_autofd int fd = -1;

  g_return_val_if_fail (MKS_IS_SCREEN (self), NULL);
  g_return_val_if_fail (!cancellable || G_IS_CANCELLABLE (cancellable), NULL);

  if (!check_console (self, error) ||
      !(paintable = _mks_paintable_new (display, cancellable, &fd, error)))
    return NULL;

  unix_fd_list = g_unix_fd_list_new_from_array (&fd, 1), fd = -1;
  if (!mks_qemu_console_call_register_listener_sync (self->console,
                                                     g_variant_new_handle (0),
                                                     unix_fd_list,
                                                     NULL,
                                                     cancellable,
                                                     error))
    return NULL;

  return g_steal_pointer (&paintable);
}
