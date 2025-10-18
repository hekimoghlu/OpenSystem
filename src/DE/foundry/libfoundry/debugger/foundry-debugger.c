/* foundry-debugger.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include "foundry-debugger.h"
#include "foundry-debugger-event.h"
#include "foundry-debugger-log-message.h"
#include "foundry-debugger-mapped-region.h"
#include "foundry-debugger-module.h"
#include "foundry-debugger-target.h"
#include "foundry-debugger-trap.h"
#include "foundry-debugger-trap-params.h"
#include "foundry-debugger-thread.h"
#include "foundry-debugger-thread-group.h"
#include "foundry-util.h"

G_DEFINE_ABSTRACT_TYPE (FoundryDebugger, foundry_debugger, FOUNDRY_TYPE_CONTEXTUAL)

enum {
  PROP_0,
  PROP_ADDRESS_SPACE,
  PROP_LOG_MESSAGES,
  PROP_MODULES,
  PROP_PRIMARY_THREAD,
  PROP_TERMINATED,
  PROP_THREADS,
  PROP_TRAPS,
  N_PROPS
};

enum {
  SIGNAL_EVENT,
  N_SIGNALS
};

static GParamSpec *properties[N_PROPS];
static guint signals[N_SIGNALS];

static void
foundry_debugger_get_property (GObject    *object,
                               guint       prop_id,
                               GValue     *value,
                               GParamSpec *pspec)
{
  FoundryDebugger *self = FOUNDRY_DEBUGGER (object);

  switch (prop_id)
    {
    case PROP_ADDRESS_SPACE:
      g_value_take_object (value, foundry_debugger_list_address_space (self));
      break;

    case PROP_LOG_MESSAGES:
      g_value_take_object (value, foundry_debugger_list_log_messages (self));
      break;

    case PROP_MODULES:
      g_value_take_object (value, foundry_debugger_list_modules (self));
      break;

    case PROP_PRIMARY_THREAD:
      g_value_take_object (value, foundry_debugger_dup_primary_thread (self));
      break;

    case PROP_TERMINATED:
      g_value_set_boolean (value, foundry_debugger_has_terminated (self));
      break;

    case PROP_THREADS:
      g_value_take_object (value, foundry_debugger_list_threads (self));
      break;

    case PROP_TRAPS:
      g_value_take_object (value, foundry_debugger_list_traps (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_class_init (FoundryDebuggerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_debugger_get_property;

  properties[PROP_ADDRESS_SPACE] =
    g_param_spec_object ("address-space", NULL, NULL,
                         G_TYPE_LIST_MODEL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_LOG_MESSAGES] =
    g_param_spec_object ("log-messages", NULL, NULL,
                         G_TYPE_LIST_MODEL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_MODULES] =
    g_param_spec_object ("modules", NULL, NULL,
                         G_TYPE_LIST_MODEL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  /**
   * FoundryDebugger:primary-thread:
   *
   * The first thread that was created by the debugger.
   *
   * Since: 1.1
   */
  properties[PROP_PRIMARY_THREAD] =
    g_param_spec_object ("primary-thread", NULL, NULL,
                         FOUNDRY_TYPE_DEBUGGER_THREAD,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  /**
   * FoundryDebugger:terminated:
   *
   * If the debuggee has terminated.
   *
   * Since: 1.1
   */
  properties[PROP_TERMINATED] =
    g_param_spec_boolean ("terminated", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  /**
   * FoundryDebugger:threads:
   *
   * Since: 1.1
   */
  properties[PROP_THREADS] =
    g_param_spec_object ("threads", NULL, NULL,
                         G_TYPE_LIST_MODEL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TRAPS] =
    g_param_spec_object ("traps", NULL, NULL,
                         G_TYPE_LIST_MODEL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  signals[SIGNAL_EVENT] =
    g_signal_new ("event",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  G_STRUCT_OFFSET (FoundryDebuggerClass, event),
                  NULL, NULL,
                  NULL,
                  G_TYPE_NONE, 1, FOUNDRY_TYPE_DEBUGGER_EVENT);
}

static void
foundry_debugger_init (FoundryDebugger *self)
{
}

/**
 * foundry_debugger_dup_name:
 * @self: a #FoundryDebugger
 *
 * Gets a name for the provider that is expected to be displayed to
 * users such as "GNU Debugger".
 *
 * Returns: (transfer full): the name of the provider
 */
char *
foundry_debugger_dup_name (FoundryDebugger *self)
{
  char *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER (self), NULL);

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->dup_name)
    ret = FOUNDRY_DEBUGGER_GET_CLASS (self)->dup_name (self);

  if (ret == NULL)
    ret = g_strdup (G_OBJECT_TYPE_NAME (self));

  return g_steal_pointer (&ret);
}

/**
 * foundry_debugger_connect_to_target:
 * @self: a [class@Foundry.Debugger]
 * @target: a [class@Foundry.DebuggerTarget]
 *
 * Connects to @target.
 *
 * Not all debuggers may not support all debugger target types.
 *
 * Returns: (transfer full): [class@Dex.Future] that resolves to any value
 *   or rejects with error.
 */
DexFuture *
foundry_debugger_connect_to_target (FoundryDebugger       *self,
                                    FoundryDebuggerTarget *target)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER (self));
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER_TARGET (target));

  return FOUNDRY_DEBUGGER_GET_CLASS (self)->connect_to_target (self, target);
}

/**
 * foundry_debugger_initialize:
 * @self: a [class@Foundry.Debugger]
 *
 * This must be called before using the debugger instance and may only
 * be called once.
 *
 * Subclasses are expected to perform capability negotiation as part
 * of this request.
 *
 * Returns: (transfer full): [class@Dex.Future] that resolves to any value
 *   or rejects with error.
 */
DexFuture *
foundry_debugger_initialize (FoundryDebugger *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER (self));

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->initialize)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->initialize (self);

  return dex_future_new_true ();
}

/**
 * foundry_debugger_list_address_space:
 * @self: a [class@Foundry.Debugger]
 *
 * Gets a [iface@Gio.ListModel] of [class@Foundry.DebuggerMappedRegion]
 * that is updated based on the address mapping of the debuggee.
 *
 * Returns: (transfer full):
 */
GListModel *
foundry_debugger_list_address_space (FoundryDebugger *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER (self), NULL);

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->list_address_space)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->list_address_space (self);

  return G_LIST_MODEL (g_list_store_new (FOUNDRY_TYPE_DEBUGGER_MAPPED_REGION));
}

/**
 * foundry_debugger_list_modules:
 * @self: a [class@Foundry.Debugger]
 *
 * Lists the known modules loaded into the address space.
 *
 * Returns: (transfer full): a [iface@Gio.ListModel] of [class@Foundry.DebuggerTrap]
 */
GListModel *
foundry_debugger_list_modules (FoundryDebugger *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER (self), NULL);

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->list_modules)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->list_modules (self);

  return G_LIST_MODEL (g_list_store_new (FOUNDRY_TYPE_DEBUGGER_MODULE));
}

/**
 * foundry_debugger_list_traps:
 * @self: a [class@Foundry.Debugger]
 *
 * List known traps (breakpoints, countpoints, watchpoints) that have been
 * registered with the debugger.
 *
 * Returns: (transfer full): a [iface@Gio.ListModel] of [class@Foundry.DebuggerTrap]
 */
GListModel *
foundry_debugger_list_traps (FoundryDebugger *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER (self), NULL);

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->list_traps)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->list_traps (self);

  return G_LIST_MODEL (g_list_store_new (FOUNDRY_TYPE_DEBUGGER_TRAP));
}

/**
 * foundry_debugger_list_threads:
 * @self: a [class@Foundry.Debugger]
 *
 * List threads known to the debugger.
 *
 * Returns: (transfer full): a [iface@Gio.ListModel] of [class@Foundry.DebuggerThread]
 */
GListModel *
foundry_debugger_list_threads (FoundryDebugger *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER (self), NULL);

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->list_threads)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->list_threads (self);

  return G_LIST_MODEL (g_list_store_new (FOUNDRY_TYPE_DEBUGGER_THREAD));
}

/**
 * foundry_debugger_list_thread_groups:
 * @self: a [class@Foundry.Debugger]
 *
 * List thread groups known to the debugger.
 *
 * Returns: (transfer full): a [iface@Gio.ListModel] of [class@Foundry.DebuggerThreadGroup]
 */
GListModel *
foundry_debugger_list_thread_groups (FoundryDebugger *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER (self), NULL);

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->list_thread_groups)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->list_thread_groups (self);

  return G_LIST_MODEL (g_list_store_new (FOUNDRY_TYPE_DEBUGGER_THREAD_GROUP));
}

/**
 * foundry_debugger_disassemble:
 * @self: a [class@Foundry.Debugger]
 * @begin_address:
 * @end_address:
 *
 * Disassemble the instructions found in the address range.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.DebuggerInstruction].
 */
DexFuture *
foundry_debugger_disassemble (FoundryDebugger *self,
                              guint64          begin_address,
                              guint64          end_address)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER (self));

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->disassemble)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->disassemble (self, begin_address, end_address);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_debugger_interrupt:
 * @self: a [class@Foundry.Debugger]
 *
 * This should cause the child process to pause.
 *
 * Use [method@Foundry.DebuggerThread.interrupt] in new code.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any
 *   value or rejects with error.
 *
 * Deprecated: 1.1
 */
DexFuture *
foundry_debugger_interrupt (FoundryDebugger *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER (self));

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->interrupt)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->interrupt (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_debugger_interpret:
 * @self: a [class@Foundry.Debugger]
 * @command: the command to interpret
 *
 * Requests that the debugger interpret a command. This is typically the
 * REPL of a debugger and can be used to bridge the normal interpreter of
 * a debugger to the the user.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any
 *   value or rejects with error.
 */
DexFuture *
foundry_debugger_interpret (FoundryDebugger *self,
                            const char      *command)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER (self));
  dex_return_error_if_fail (command != NULL);

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->interpret)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->interpret (self, command);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_debugger_send_signal:
 * @self: a [class@Foundry.Debugger]
 *
 * Send signal @signum to debuggee.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value or rejects with error.
 */
DexFuture *
foundry_debugger_send_signal (FoundryDebugger *self,
                              int              signum)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER (self));

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->send_signal)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->send_signal (self, signum);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_debugger_stop:
 * @self: a [class@Foundry.Debugger]
 *
 * Stop the debugger fully. This should at least cause the inferior to be
 * sent a terminating signal.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value or rejects with error.
 */
DexFuture *
foundry_debugger_stop (FoundryDebugger *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER (self));

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->stop)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->stop (self);

  return foundry_debugger_send_signal (self, SIGKILL);
}

/**
 * foundry_debugger_can_move:
 * @self: a [class@Foundry.Debugger]
 *
 * Determines of the debugger can currently make @movement.
 *
 * Use [method@Foundry.DebuggerThread.can_move] in new code.
 *
 * Returns: %TRUE if @movement can be performed
 *
 * Deprecated: 1.1
 */
gboolean
foundry_debugger_can_move (FoundryDebugger         *self,
                           FoundryDebuggerMovement  movement)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER (self), FALSE);

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->can_move)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->can_move (self, movement);

  return FALSE;
}

/**
 * foundry_debugger_move:
 * @self: a [class@Foundry.Debugger]
 * @movement: how to move within the debugger
 *
 * Use [method@Foundry.DebuggerThread.move] in new code.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value or rejects with error.
 *
 * Deprecated: 1.1
 */
DexFuture *
foundry_debugger_move (FoundryDebugger         *self,
                       FoundryDebuggerMovement  movement)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER (self));

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->move)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->move (self, movement);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_debugger_trap:
 * @self: a [class@Foundry.Debugger]
 * @params: the params for creating the new trap
 *
 * Register a new breakpoint based on @params.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value or rejects with error.
 */
DexFuture *
foundry_debugger_trap (FoundryDebugger           *self,
                       FoundryDebuggerTrapParams *params)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER (self));
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER_TRAP_PARAMS (params));

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->trap)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->trap (self, params);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_debugger_emit_event:
 *
 * This should only be used by debugger implementations.
 */
void
foundry_debugger_emit_event (FoundryDebugger      *self,
                             FoundryDebuggerEvent *event)
{
  g_return_if_fail (FOUNDRY_IS_DEBUGGER (self));
  g_return_if_fail (FOUNDRY_IS_DEBUGGER_EVENT (event));

  g_signal_emit (self, signals[SIGNAL_EVENT], 0, event);
}

/**
 * foundry_debugger_list_log_messages:
 * @self: a [class@Foundry.Debugger]
 *
 * Lists available log messages from the debugger instance.
 *
 * Returns: (transfer full): a [iface@Gio.ListModel] of
 *   [class@Foundry.DebuggerLogMessage].
 *
 * Since: 1.1
 */
GListModel *
foundry_debugger_list_log_messages (FoundryDebugger *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER (self), NULL);

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->list_log_messages)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->list_log_messages (self);

  return G_LIST_MODEL (g_list_store_new (FOUNDRY_TYPE_DEBUGGER_LOG_MESSAGE));
}

/**
 * foundry_debugger_dup_primary_thread:
 * @self: a [class@Foundry.Debugger]
 *
 * Gets a copy of the primary thread (the first thread created by the debugger).
 *
 * Returns: (transfer full) (nullable): a [class@Foundry.DebuggerThread] or %NULL
 *
 * Since: 1.1
 */
FoundryDebuggerThread *
foundry_debugger_dup_primary_thread (FoundryDebugger *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER (self), NULL);

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->dup_primary_thread)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->dup_primary_thread (self);

  return NULL;
}

gboolean
foundry_debugger_has_terminated (FoundryDebugger *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER (self), FALSE);

  if (FOUNDRY_DEBUGGER_GET_CLASS (self)->has_terminated)
    return FOUNDRY_DEBUGGER_GET_CLASS (self)->has_terminated (self);

  return FALSE;
}

G_DEFINE_ENUM_TYPE (FoundryDebuggerMovement, foundry_debugger_movement,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_MOVEMENT_START, "start"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_MOVEMENT_CONTINUE, "continue"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_MOVEMENT_STEP_IN, "step-in"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_MOVEMENT_STEP_OUT, "step-out"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_DEBUGGER_MOVEMENT_STEP_OVER, "step-over"))
