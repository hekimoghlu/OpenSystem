/* foundry-responder.c
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

#include "foundry-responder-private.h"

typedef struct
{
  FoundryReaction *reaction;
} FoundryResponderPrivate;

enum {
  PROP_0,
  PROP_REACTION,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryResponder, foundry_responder, FOUNDRY_TYPE_REACTION)

static GParamSpec *properties[N_PROPS];

static void
foundry_responder_react (FoundryReaction *reaction)
{
  FoundryResponder *self = (FoundryResponder *)reaction;
  FoundryResponderPrivate *priv = foundry_responder_get_instance_private (self);

  g_assert (FOUNDRY_IS_RESPONDER (self));

  if (priv->reaction)
    foundry_reaction_react (priv->reaction);
}

static void
foundry_responder_finalize (GObject *object)
{
  FoundryResponder *self = (FoundryResponder *)object;
  FoundryResponderPrivate *priv = foundry_responder_get_instance_private (self);

  g_clear_object (&priv->reaction);

  G_OBJECT_CLASS (foundry_responder_parent_class)->finalize (object);
}

static void
foundry_responder_get_property (GObject    *object,
                                guint       prop_id,
                                GValue     *value,
                                GParamSpec *pspec)
{
  FoundryResponder *self = FOUNDRY_RESPONDER (object);

  switch (prop_id)
    {
    case PROP_REACTION:
      g_value_set_object (value, foundry_responder_get_reaction (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_responder_set_property (GObject      *object,
                                guint         prop_id,
                                const GValue *value,
                                GParamSpec   *pspec)
{
  FoundryResponder *self = FOUNDRY_RESPONDER (object);

  switch (prop_id)
    {
    case PROP_REACTION:
      foundry_responder_set_reaction (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_responder_class_init (FoundryResponderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryReactionClass *reaction_class = FOUNDRY_REACTION_CLASS (klass);

  object_class->finalize = foundry_responder_finalize;
  object_class->get_property = foundry_responder_get_property;
  object_class->set_property = foundry_responder_set_property;

  reaction_class->react = foundry_responder_react;

  properties[PROP_REACTION] =
    g_param_spec_object ("reaction", NULL, NULL,
                         FOUNDRY_TYPE_REACTION,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_responder_init (FoundryResponder *self)
{
}

/**
 * foundry_responder_get_reaction:
 * @self: a [class@Foundry.Responder]
 *
 * Returns: (transfer none) (nullable):
 */
FoundryReaction *
foundry_responder_get_reaction (FoundryResponder *self)
{
  FoundryResponderPrivate *priv = foundry_responder_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_RESPONDER (self), NULL);

  return priv->reaction;
}

void
foundry_responder_set_reaction (FoundryResponder *self,
                                FoundryReaction  *reaction)
{
  FoundryResponderPrivate *priv = foundry_responder_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_RESPONDER (self));
  g_return_if_fail (!reaction || FOUNDRY_IS_REACTION (reaction));

  if (g_set_object (&priv->reaction, reaction))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_REACTION]);
}
