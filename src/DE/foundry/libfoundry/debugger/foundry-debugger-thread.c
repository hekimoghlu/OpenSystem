/* foundry-debugger-thread.c
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "foundry-debugger-thread.h"
#include "foundry-util.h"

enum {
  PROP_0,
  PROP_ID,
  PROP_GROUP_ID,
  PROP_STOPPED,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryDebuggerThread, foundry_debugger_thread, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_debugger_thread_get_property (GObject    *object,
                                      guint       prop_id,
                                      GValue     *value,
                                      GParamSpec *pspec)
{
  FoundryDebuggerThread *self = FOUNDRY_DEBUGGER_THREAD (object);

  switch (prop_id)
    {
    case PROP_ID:
      g_value_take_string (value, foundry_debugger_thread_dup_id (self));
      break;

    case PROP_GROUP_ID:
      g_value_take_string (value, foundry_debugger_thread_dup_group_id (self));
      break;

    case PROP_STOPPED:
      g_value_set_boolean (value, foundry_debugger_thread_is_stopped (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_thread_class_init (FoundryDebuggerThreadClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_debugger_thread_get_property;

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_GROUP_ID] =
    g_param_spec_string ("group-id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  /**
   * FoundryDebuggerThread:stopped:
   *
   * If the thread is known to be stopped.
   *
   * Since: 1.1
   */
  properties[PROP_STOPPED] =
    g_param_spec_boolean ("stopped", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_debugger_thread_init (FoundryDebuggerThread *self)
{
}

char *
foundry_debugger_thread_dup_id (FoundryDebuggerThread *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_THREAD (self), NULL);

  if (FOUNDRY_DEBUGGER_THREAD_GET_CLASS (self)->dup_id)
    return FOUNDRY_DEBUGGER_THREAD_GET_CLASS (self)->dup_id (self);

  return NULL;
}

char *
foundry_debugger_thread_dup_group_id (FoundryDebuggerThread *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_THREAD (self), NULL);

  if (FOUNDRY_DEBUGGER_THREAD_GET_CLASS (self)->dup_group_id)
    return FOUNDRY_DEBUGGER_THREAD_GET_CLASS (self)->dup_group_id (self);

  return NULL;
}

/**
 * foundry_debugger_thread_list_frames:
 * @self: a [class@Foundry.DebuggerThread]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.DebuggerStackFrame].
 */
DexFuture *
foundry_debugger_thread_list_frames (FoundryDebuggerThread *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER_THREAD (self));

  if (FOUNDRY_DEBUGGER_THREAD_GET_CLASS (self)->list_frames)
    return FOUNDRY_DEBUGGER_THREAD_GET_CLASS (self)->list_frames (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_debugger_thread_is_stopped:
 * @self: a [class@Foundry.DebuggerThread]
 *
 * Gets if the thread is currently stopped.
 *
 * Subclasses should [method@GObject.Object.notify] the "stopped" property
 * when this value changes.
 *
 * Returns: %TRUE if thread is stopped
 *
 * Since: 1.1
 */
gboolean
foundry_debugger_thread_is_stopped (FoundryDebuggerThread *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_THREAD (self), FALSE);

  if (FOUNDRY_DEBUGGER_THREAD_GET_CLASS (self)->is_stopped)
    return FOUNDRY_DEBUGGER_THREAD_GET_CLASS (self)->is_stopped (self);

  return FALSE;
}

/**
 * foundry_debugger_thread_move:
 * @self: a [class@Foundry.DebuggerThread]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value or rejects with error.
 *
 * Since: 1.1
 */
DexFuture *
foundry_debugger_thread_move (FoundryDebuggerThread   *self,
                              FoundryDebuggerMovement  movement)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER_THREAD (self));

  if (FOUNDRY_DEBUGGER_THREAD_GET_CLASS (self)->move)
    return FOUNDRY_DEBUGGER_THREAD_GET_CLASS (self)->move (self, movement);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_debugger_thread_can_move:
 *
 * If @self can attempt @movement at this point.
 *
 * Returns: %TRUE if @movement may be a valid movement now.
 */
gboolean
foundry_debugger_thread_can_move (FoundryDebuggerThread   *self,
                                  FoundryDebuggerMovement  movement)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_THREAD (self), FALSE);

  if (FOUNDRY_DEBUGGER_THREAD_GET_CLASS (self)->can_move)
    return FOUNDRY_DEBUGGER_THREAD_GET_CLASS (self)->can_move (self, movement);

  return FALSE;
}

/**
 * foundry_debugger_thread_interrupt:
 * @self: a [class@Foundry.DebuggerThread]
 *
 * Stops the thread.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any value
 *   or rejects with error.
 *
 * Since: 1.1
 */
DexFuture *
foundry_debugger_thread_interrupt (FoundryDebuggerThread *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER_THREAD (self));

  if (FOUNDRY_DEBUGGER_THREAD_GET_CLASS (self)->interrupt)
    return FOUNDRY_DEBUGGER_THREAD_GET_CLASS (self)->interrupt (self);

  return foundry_future_new_not_supported ();
}
