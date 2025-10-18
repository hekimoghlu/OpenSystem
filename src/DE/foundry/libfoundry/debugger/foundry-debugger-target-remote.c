/* foundry-debugger-target-remote.c
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

#include "foundry-debugger-target-remote.h"
#include "foundry-debugger-target-private.h"

struct _FoundryDebuggerTargetRemote
{
  FoundryDebuggerTarget parent_instance;
  char *address;
};

struct _FoundryDebuggerTargetRemoteClass
{
  FoundryDebuggerTargetClass parent_class;
};

G_DEFINE_FINAL_TYPE (FoundryDebuggerTargetRemote, foundry_debugger_target_remote, FOUNDRY_TYPE_DEBUGGER_TARGET)

enum {
  PROP_0,
  PROP_ADDRESS,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_debugger_target_remote_finalize (GObject *object)
{
  FoundryDebuggerTargetRemote *self = (FoundryDebuggerTargetRemote *)object;

  g_clear_pointer (&self->address, g_free);

  G_OBJECT_CLASS (foundry_debugger_target_remote_parent_class)->finalize (object);
}

static void
foundry_debugger_target_remote_get_property (GObject    *object,
                                             guint       prop_id,
                                             GValue     *value,
                                             GParamSpec *pspec)
{
  FoundryDebuggerTargetRemote *self = FOUNDRY_DEBUGGER_TARGET_REMOTE (object);

  switch (prop_id)
    {
    case PROP_ADDRESS:
      g_value_take_string (value, foundry_debugger_target_remote_dup_address (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_target_remote_set_property (GObject      *object,
                                             guint         prop_id,
                                             const GValue *value,
                                             GParamSpec   *pspec)
{
  FoundryDebuggerTargetRemote *self = FOUNDRY_DEBUGGER_TARGET_REMOTE (object);

  switch (prop_id)
    {
    case PROP_ADDRESS:
      self->address = g_value_dup_string (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_target_remote_class_init (FoundryDebuggerTargetRemoteClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_debugger_target_remote_finalize;
  object_class->get_property = foundry_debugger_target_remote_get_property;
  object_class->set_property = foundry_debugger_target_remote_set_property;

  properties[PROP_ADDRESS] =
    g_param_spec_string ("address", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_debugger_target_remote_init (FoundryDebuggerTargetRemote *self)
{
}

/**
 * foundry_debugger_target_remote_dup_address:
 * @self: a [class@Foundry.DebuggerTargetCommand]
 *
 */
char *
foundry_debugger_target_remote_dup_address (FoundryDebuggerTargetRemote *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_TARGET_REMOTE (self), NULL);

  return g_strdup (self->address);
}

