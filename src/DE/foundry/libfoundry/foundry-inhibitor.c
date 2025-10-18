/* foundry-inhibitor.c
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

#include "foundry-context-private.h"
#include "foundry-contextual.h"
#include "foundry-inhibitor-private.h"

struct _FoundryInhibitor
{
  GObject         parent_instance;
  FoundryContext *context;
  guint           inhibited : 1;
};

enum {
  PROP_0,
  PROP_CONTEXT,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryInhibitor, foundry_inhibitor, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_inhibitor_dispose (GObject *object)
{
  FoundryInhibitor *self = (FoundryInhibitor *)object;

  if (self->inhibited)
    foundry_inhibitor_uninhibit (self);

  g_clear_object (&self->context);

  G_OBJECT_CLASS (foundry_inhibitor_parent_class)->dispose (object);
}

static void
foundry_inhibitor_get_property (GObject    *object,
                                guint       prop_id,
                                GValue     *value,
                                GParamSpec *pspec)
{
  FoundryInhibitor *self = FOUNDRY_INHIBITOR (object);

  switch (prop_id)
    {
    case PROP_CONTEXT:
      g_value_take_object (value, foundry_inhibitor_dup_context (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_inhibitor_set_property (GObject      *object,
                                guint         prop_id,
                                const GValue *value,
                                GParamSpec   *pspec)
{
  FoundryInhibitor *self = FOUNDRY_INHIBITOR (object);

  switch (prop_id)
    {
    case PROP_CONTEXT:
      self->context = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_inhibitor_class_init (FoundryInhibitorClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_inhibitor_dispose;
  object_class->get_property = foundry_inhibitor_get_property;
  object_class->set_property = foundry_inhibitor_set_property;

  properties[PROP_CONTEXT] =
    g_param_spec_object ("context", NULL, NULL,
                         FOUNDRY_TYPE_CONTEXT,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_inhibitor_init (FoundryInhibitor *self)
{
}

/**
 * foundry_inhibitor_dup_context:
 * @self: a #FoundryInhibitor
 *
 * Returns: (transfer full): a #FoundryContext or %NULL
 */
FoundryContext *
foundry_inhibitor_dup_context (FoundryInhibitor *self)
{
  g_return_val_if_fail (FOUNDRY_IS_INHIBITOR (self), NULL);

  return self->context ? g_object_ref (self->context) : NULL;
}

void
foundry_inhibitor_uninhibit (FoundryInhibitor *self)
{
  g_return_if_fail (FOUNDRY_IS_INHIBITOR (self));
  g_return_if_fail (!self->inhibited || FOUNDRY_IS_CONTEXT (self->context));

  if (self->inhibited)
    {
      self->inhibited = FALSE;
      _foundry_context_uninhibit (self->context);
    }
}

FoundryInhibitor *
foundry_inhibitor_new (FoundryContext  *context,
                       GError         **error)
{
  FoundryInhibitor *self;

  g_assert (!context || FOUNDRY_IS_CONTEXT (context));

  if (!context || !_foundry_context_inhibit (context))
    {
      g_set_error (error,
                   FOUNDRY_CONTEXTUAL_ERROR,
                   FOUNDRY_CONTEXTUAL_ERROR_IN_SHUTDOWN,
                   "Context is already in shutdown");
      return FALSE;
    }

  self = g_object_new (FOUNDRY_TYPE_INHIBITOR,
                       "context", context,
                       NULL);

  self->inhibited = TRUE;

  return self;
}
