/* foundry-debugger-actions.c
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

#include "foundry-debugger.h"
#include "foundry-debugger-actions.h"
#include "foundry-debugger-thread.h"
#include "foundry-util.h"

#include "eggactiongroup.h"

struct _FoundryDebuggerActions
{
  GObject                parent_instance;
  FoundryDebugger       *debugger;
  FoundryDebuggerThread *thread;
  gulong                 thread_changed_id;
  gulong                 notify_terminated_id;
};

enum {
  PROP_0,
  PROP_DEBUGGER,
  PROP_THREAD,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
do_move (FoundryDebuggerActions  *self,
         FoundryDebuggerMovement  movement)
{
  G_GNUC_BEGIN_IGNORE_DEPRECATIONS

  if (self->thread != NULL)
    dex_future_disown (foundry_debugger_thread_move (self->thread, movement));
  else if (self->debugger != NULL)
    dex_future_disown (foundry_debugger_move (self->debugger, movement));

  G_GNUC_END_IGNORE_DEPRECATIONS
}

static void
foundry_debugger_actions_continue_action (FoundryDebuggerActions *self,
                                          GVariant               *param)
{
  g_assert (FOUNDRY_IS_DEBUGGER_ACTIONS (self));

  do_move (self, FOUNDRY_DEBUGGER_MOVEMENT_CONTINUE);
}

static void
foundry_debugger_actions_step_in_action (FoundryDebuggerActions *self,
                                         GVariant               *param)
{
  g_assert (FOUNDRY_IS_DEBUGGER_ACTIONS (self));

  do_move (self, FOUNDRY_DEBUGGER_MOVEMENT_STEP_IN);
}

static void
foundry_debugger_actions_step_out_action (FoundryDebuggerActions *self,
                                          GVariant               *param)
{
  g_assert (FOUNDRY_IS_DEBUGGER_ACTIONS (self));

  do_move (self, FOUNDRY_DEBUGGER_MOVEMENT_STEP_OUT);
}

static void
foundry_debugger_actions_step_over_action (FoundryDebuggerActions *self,
                                           GVariant               *param)
{
  g_assert (FOUNDRY_IS_DEBUGGER_ACTIONS (self));

  do_move (self, FOUNDRY_DEBUGGER_MOVEMENT_STEP_OVER);
}

static void
foundry_debugger_actions_interrupt_action (FoundryDebuggerActions *self,
                                           GVariant               *param)
{
  g_assert (FOUNDRY_IS_DEBUGGER_ACTIONS (self));

  G_GNUC_BEGIN_IGNORE_DEPRECATIONS

  if (self->thread != NULL)
    dex_future_disown (foundry_debugger_thread_interrupt (self->thread));
  else if (self->debugger != NULL)
    dex_future_disown (foundry_debugger_interrupt (self->debugger));

  G_GNUC_END_IGNORE_DEPRECATIONS
}

static void
foundry_debugger_actions_stop (FoundryDebuggerActions *self,
                               GVariant               *param)
{
  g_assert (FOUNDRY_IS_DEBUGGER_ACTIONS (self));

  dex_future_disown (foundry_debugger_stop (self->debugger));
}

EGG_DEFINE_ACTION_GROUP (FoundryDebuggerActions, foundry_debugger_actions, {
  { "continue", foundry_debugger_actions_continue_action, NULL, NULL },
  { "step-in", foundry_debugger_actions_step_in_action, NULL, NULL },
  { "step-out", foundry_debugger_actions_step_out_action, NULL, NULL },
  { "step-over", foundry_debugger_actions_step_over_action, NULL, NULL },
  { "interrupt", foundry_debugger_actions_interrupt_action, NULL, NULL },
  { "stop", foundry_debugger_actions_stop, NULL, NULL },
})

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryDebuggerActions, foundry_debugger_actions, G_TYPE_OBJECT,
                                EGG_IMPLEMENT_ACTION_GROUP (foundry_debugger_actions))

static void
foundry_debugger_actions_update (FoundryDebuggerActions *self)
{
  gboolean is_stopped = TRUE;
  gboolean has_terminated = FALSE;
  gboolean can_continue = FALSE;
  gboolean can_step_in = FALSE;
  gboolean can_step_out = FALSE;
  gboolean can_step_over = FALSE;

  g_assert (!self->thread || FOUNDRY_IS_DEBUGGER_THREAD (self->thread));
  g_assert (!self->debugger || FOUNDRY_IS_DEBUGGER (self->debugger));
  g_assert (FOUNDRY_IS_DEBUGGER_ACTIONS (self));

  if (self->debugger != NULL && self->thread != NULL)
    is_stopped = foundry_debugger_thread_is_stopped (self->thread);

  if (self->debugger != NULL)
    has_terminated = foundry_debugger_has_terminated (self->debugger);

  if (self->thread != NULL && !has_terminated && is_stopped)
    {
      can_continue = foundry_debugger_thread_can_move (self->thread, FOUNDRY_DEBUGGER_MOVEMENT_CONTINUE);
      can_step_in = foundry_debugger_thread_can_move (self->thread, FOUNDRY_DEBUGGER_MOVEMENT_STEP_IN);
      can_step_out = foundry_debugger_thread_can_move (self->thread, FOUNDRY_DEBUGGER_MOVEMENT_STEP_OUT);
      can_step_over = foundry_debugger_thread_can_move (self->thread, FOUNDRY_DEBUGGER_MOVEMENT_STEP_OVER);
    }

  foundry_debugger_actions_set_action_enabled (self, "continue", can_continue);
  foundry_debugger_actions_set_action_enabled (self, "step-in", can_step_in);
  foundry_debugger_actions_set_action_enabled (self, "step-out", can_step_out);
  foundry_debugger_actions_set_action_enabled (self, "step-over", can_step_over);
  foundry_debugger_actions_set_action_enabled (self, "interrupt", !has_terminated && !is_stopped);
  foundry_debugger_actions_set_action_enabled (self, "stop", !has_terminated);
}

static void
foundry_debugger_actions_thread_changed_cb (FoundryDebuggerActions *self,
                                            GParamSpec             *pspec,
                                            FoundryDebuggerThread  *thread)
{
  g_assert (FOUNDRY_IS_DEBUGGER_ACTIONS (self));
  g_assert (FOUNDRY_IS_DEBUGGER_THREAD (thread));

  foundry_debugger_actions_update (self);
}

static void
foundry_debugger_actions_constructed (GObject *object)
{
  FoundryDebuggerActions *self = (FoundryDebuggerActions *)object;

  G_OBJECT_CLASS (foundry_debugger_actions_parent_class)->constructed (object);

  if (self->debugger != NULL)
    g_signal_connect_object (self->debugger,
                             "notify::terminated",
                             G_CALLBACK (foundry_debugger_actions_update),
                             self,
                             G_CONNECT_SWAPPED);
}

static void
foundry_debugger_actions_dispose (GObject *object)
{
  FoundryDebuggerActions *self = FOUNDRY_DEBUGGER_ACTIONS (object);

  g_clear_signal_handler (&self->thread_changed_id, self->thread);
  g_clear_signal_handler (&self->notify_terminated_id, self->debugger);

  g_clear_object (&self->thread);
  g_clear_object (&self->debugger);

  G_OBJECT_CLASS (foundry_debugger_actions_parent_class)->dispose (object);
}

static void
foundry_debugger_actions_get_property (GObject    *object,
                                       guint       prop_id,
                                       GValue     *value,
                                       GParamSpec *pspec)
{
  FoundryDebuggerActions *self = FOUNDRY_DEBUGGER_ACTIONS (object);

  switch (prop_id)
    {
    case PROP_DEBUGGER:
      g_value_take_object (value, foundry_debugger_actions_dup_debugger (self));
      break;

    case PROP_THREAD:
      g_value_take_object (value, foundry_debugger_actions_dup_thread (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_actions_set_property (GObject      *object,
                                       guint         prop_id,
                                       const GValue *value,
                                       GParamSpec   *pspec)
{
  FoundryDebuggerActions *self = FOUNDRY_DEBUGGER_ACTIONS (object);

  switch (prop_id)
    {
    case PROP_DEBUGGER:
      foundry_debugger_actions_set_debugger (self, g_value_get_object (value));
      break;

    case PROP_THREAD:
      foundry_debugger_actions_set_thread (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_actions_class_init (FoundryDebuggerActionsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->constructed = foundry_debugger_actions_constructed;
  object_class->get_property = foundry_debugger_actions_get_property;
  object_class->set_property = foundry_debugger_actions_set_property;
  object_class->dispose = foundry_debugger_actions_dispose;

  /**
   * FoundryDebuggerActions:debugger:
   *
   * The debugger instance.
   *
   * Since: 1.1
   */
  properties[PROP_DEBUGGER] =
    g_param_spec_object ("debugger", NULL, NULL,
                         FOUNDRY_TYPE_DEBUGGER,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  /**
   * FoundryDebuggerActions:thread:
   *
   * The debugger thread.
   *
   * Since: 1.1
   */
  properties[PROP_THREAD] =
    g_param_spec_object ("thread", NULL, NULL,
                         FOUNDRY_TYPE_DEBUGGER_THREAD,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_debugger_actions_init (FoundryDebuggerActions *self)
{
}

/**
 * foundry_debugger_actions_new:
 * @debugger: (nullable): a [class@Foundry.Debugger]
 * @thread: (nullable): a [class@Foundry.DebuggerThread]
 *
 * Creates a new [class@Foundry.DebuggerActions] instance.
 *
 * Returns: (transfer full): a new [class@Foundry.DebuggerActions]
 *
 * Since: 1.1
 */
FoundryDebuggerActions *
foundry_debugger_actions_new (FoundryDebugger       *debugger,
                              FoundryDebuggerThread *thread)
{
  g_return_val_if_fail (!debugger || FOUNDRY_IS_DEBUGGER (debugger), NULL);
  g_return_val_if_fail (!thread || FOUNDRY_IS_DEBUGGER_THREAD (thread), NULL);

  return g_object_new (FOUNDRY_TYPE_DEBUGGER_ACTIONS,
                       "debugger", debugger,
                       "thread", thread,
                       NULL);
}

/**
 * foundry_debugger_actions_dup_debugger:
 * @self: a [class@Foundry.DebuggerActions]
 *
 * Gets the debugger instance.
 *
 * Returns: (transfer full) (nullable): the debugger instance
 *
 * Since: 1.1
 */
FoundryDebugger *
foundry_debugger_actions_dup_debugger (FoundryDebuggerActions *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_ACTIONS (self), NULL);

  return self->debugger ? g_object_ref (self->debugger) : NULL;
}

/**
 * foundry_debugger_actions_set_debugger:
 * @self: a [class@Foundry.DebuggerActions]
 * @debugger: (nullable): the debugger instance
 *
 * Sets the debugger instance.
 *
 * Since: 1.1
 */
void
foundry_debugger_actions_set_debugger (FoundryDebuggerActions *self,
                                       FoundryDebugger        *debugger)
{
  g_return_if_fail (FOUNDRY_IS_DEBUGGER_ACTIONS (self));
  g_return_if_fail (debugger == NULL || FOUNDRY_IS_DEBUGGER (debugger));

  if (self->debugger == debugger)
    return;

  if (debugger != NULL)
    g_object_ref (debugger);

  g_clear_signal_handler (&self->notify_terminated_id, self->debugger);

  g_clear_object (&self->debugger);

  self->debugger = debugger;

  if (debugger != NULL)
    self->notify_terminated_id =
      g_signal_connect_object (debugger,
                               "notify::terminated",
                               G_CALLBACK (foundry_debugger_actions_update),
                               self,
                               G_CONNECT_SWAPPED);

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_DEBUGGER]);

  foundry_debugger_actions_update (self);
}

/**
 * foundry_debugger_actions_dup_thread:
 * @self: a [class@Foundry.DebuggerActions]
 *
 * Gets the debugger thread.
 *
 * Returns: (transfer full) (nullable): the debugger thread
 *
 * Since: 1.1
 */
FoundryDebuggerThread *
foundry_debugger_actions_dup_thread (FoundryDebuggerActions *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_ACTIONS (self), NULL);

  return self->thread ? g_object_ref (self->thread) : NULL;
}

/**
 * foundry_debugger_actions_set_thread:
 * @self: a [class@Foundry.DebuggerActions]
 * @thread: (nullable): the debugger thread
 *
 * Sets the debugger thread and connects to its "changed" signal
 * to update action states.
 *
 * Since: 1.1
 */
void
foundry_debugger_actions_set_thread (FoundryDebuggerActions *self,
                                     FoundryDebuggerThread  *thread)
{
  g_return_if_fail (FOUNDRY_IS_DEBUGGER_ACTIONS (self));
  g_return_if_fail (thread == NULL || FOUNDRY_IS_DEBUGGER_THREAD (thread));

  if (self->thread == thread)
    return;

  if (thread)
    g_object_ref (thread);

  g_clear_signal_handler (&self->thread_changed_id, self->thread);
  g_clear_object (&self->thread);
  self->thread = thread;

  if (self->thread != NULL)
    self->thread_changed_id = g_signal_connect_object (self->thread,
                                                       "notify::stopped",
                                                       G_CALLBACK (foundry_debugger_actions_thread_changed_cb),
                                                       self,
                                                       G_CONNECT_SWAPPED);

  foundry_debugger_actions_update (self);

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_THREAD]);
}
