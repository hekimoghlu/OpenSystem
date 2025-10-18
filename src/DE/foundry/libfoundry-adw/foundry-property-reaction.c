/* foundry-property-reaction.c
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

#include "foundry-property-reaction-private.h"

struct _FoundryPropertyReaction
{
  FoundryReaction    parent_instance;
  GWeakRef           object_wr;
  char              *name;
  GtkExpression     *expression;
};

enum {
  PROP_0,
  PROP_OBJECT,
  PROP_NAME,
  PROP_VALUE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryPropertyReaction, foundry_property_reaction, FOUNDRY_TYPE_REACTION)

static GParamSpec *properties[N_PROPS];

static void
foundry_property_reaction_react (FoundryReaction *reaction)
{
  FoundryPropertyReaction *self = (FoundryPropertyReaction *)reaction;
  GObject *object;

  g_assert (FOUNDRY_IS_PROPERTY_REACTION (self));

  if (self->name == NULL)
    return;

  if ((object = g_weak_ref_get (&self->object_wr)))
    {
      GObjectClass *type_class = G_OBJECT_GET_CLASS (object);
      GParamSpec *pspec = g_object_class_find_property (type_class, self->name);
      GValue value = G_VALUE_INIT;

      if G_UNLIKELY (pspec == NULL)
        {
          g_warning ("Object %s does not have a property named %s",
                     G_OBJECT_TYPE_NAME (object), self->name);
          goto cleanup;
        }

      if (self->expression != NULL)
        {
          GType expression_type = gtk_expression_get_value_type (self->expression);

          if (pspec->value_type == expression_type)
            {
              gtk_expression_evaluate (self->expression, NULL, &value);
            }
          else if (g_value_type_compatible (pspec->value_type, expression_type))
            {
              GValue intermediate = G_VALUE_INIT;

              if (gtk_expression_evaluate (self->expression, NULL, &intermediate))
                {
                  g_value_init (&value, pspec->value_type);
                  g_value_copy (&intermediate, &value);
                }

              g_value_unset (&intermediate);
            }
          else
            {
              g_warning ("Type %s cannot be transformed to %s",
                         g_type_name (expression_type),
                         g_type_name (pspec->value_type));
            }

          /* Use default value if nothing was provided */
          if (!G_IS_VALUE (&value))
            g_value_init (&value, pspec->value_type);

          g_object_set_property (object, pspec->name, &value);
        }

    cleanup:
      g_value_unset (&value);
      g_object_unref (object);
    }
}

static void
foundry_property_reaction_finalize (GObject *object)
{
  FoundryPropertyReaction *self = (FoundryPropertyReaction *)object;

  g_weak_ref_clear (&self->object_wr);
  g_clear_pointer (&self->name, g_free);
  g_clear_pointer (&self->expression, gtk_expression_unref);

  G_OBJECT_CLASS (foundry_property_reaction_parent_class)->finalize (object);
}

static void
foundry_property_reaction_get_property (GObject    *object,
                                        guint       prop_id,
                                        GValue     *value,
                                        GParamSpec *pspec)
{
  FoundryPropertyReaction *self = FOUNDRY_PROPERTY_REACTION (object);

  switch (prop_id)
    {
    case PROP_OBJECT:
      g_value_take_object (value, foundry_property_reaction_dup_object (self));
      break;

    case PROP_NAME:
      g_value_set_string (value, foundry_property_reaction_get_name (self));
      break;

    case PROP_VALUE:
      gtk_value_set_expression (value, foundry_property_reaction_get_value (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_property_reaction_set_property (GObject      *object,
                                        guint         prop_id,
                                        const GValue *value,
                                        GParamSpec   *pspec)
{
  FoundryPropertyReaction *self = FOUNDRY_PROPERTY_REACTION (object);

  switch (prop_id)
    {
    case PROP_OBJECT:
      foundry_property_reaction_set_object (self, g_value_get_object (value));
      break;

    case PROP_NAME:
      foundry_property_reaction_set_name (self, g_value_get_string (value));
      break;

    case PROP_VALUE:
      foundry_property_reaction_set_value (self, gtk_value_get_expression (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_property_reaction_class_init (FoundryPropertyReactionClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryReactionClass *reaction_class = FOUNDRY_REACTION_CLASS (klass);

  object_class->finalize = foundry_property_reaction_finalize;
  object_class->get_property = foundry_property_reaction_get_property;
  object_class->set_property = foundry_property_reaction_set_property;

  reaction_class->react = foundry_property_reaction_react;

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_OBJECT] =
    g_param_spec_object ("object", NULL, NULL,
                         G_TYPE_OBJECT,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_VALUE] =
    gtk_param_spec_expression ("value", NULL, NULL,
                               (G_PARAM_READWRITE |
                                G_PARAM_EXPLICIT_NOTIFY |
                                G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_property_reaction_init (FoundryPropertyReaction *self)
{
  g_weak_ref_init (&self->object_wr, NULL);
}

FoundryPropertyReaction *
foundry_property_reaction_new (GObject       *object,
                               const char    *name,
                               GtkExpression *value)
{
  g_return_val_if_fail (G_IS_OBJECT (object), NULL);
  g_return_val_if_fail (!value || GTK_IS_EXPRESSION (value), NULL);

  return g_object_new (FOUNDRY_TYPE_PROPERTY_REACTION,
                       "object", object,
                       "name", name,
                       "value", value,
                       NULL);
}

static void
foundry_property_reaction_disconnect (FoundryPropertyReaction *self,
                                      GObject                 *object)
{
  g_assert (FOUNDRY_IS_PROPERTY_REACTION (self));
  g_assert (!object || G_IS_OBJECT (object));

  if (object == NULL)
    return;
}

static void
foundry_property_reaction_connect (FoundryPropertyReaction *self,
                                   GObject                 *object)
{
  g_assert (FOUNDRY_IS_PROPERTY_REACTION (self));
  g_assert (!object || G_IS_OBJECT (object));

  if (object == NULL)
    return;
}

/**
 * foundry_property_reaction_dup_object:
 * @self: a #FoundryPropertyReaction
 *
 * Returns: (transfer full) (nullable):
 */
GObject *
foundry_property_reaction_dup_object (FoundryPropertyReaction *self)
{
  g_return_val_if_fail (FOUNDRY_IS_PROPERTY_REACTION (self), NULL);

  return g_weak_ref_get (&self->object_wr);
}

void
foundry_property_reaction_set_object (FoundryPropertyReaction *self,
                                      GObject                 *object)
{
  GObject *old_object;

  g_return_if_fail (FOUNDRY_IS_PROPERTY_REACTION (self));
  g_return_if_fail (!object || G_IS_OBJECT (object));

  old_object = g_weak_ref_get (&self->object_wr);

  if (old_object != object)
    {
      if (old_object != NULL)
        foundry_property_reaction_disconnect (self, old_object);

      g_weak_ref_set (&self->object_wr, object);

      if (object != NULL)
        foundry_property_reaction_connect (self, object);

      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_OBJECT]);
    }

  g_clear_object (&old_object);
}

/**
 * foundry_property_reaction_get_value:
 * @self: a #FoundryPropertyReaction
 *
 * Returns: (transfer none) (nullable):
 */
GtkExpression *
foundry_property_reaction_get_value (FoundryPropertyReaction *self)
{
  g_return_val_if_fail (FOUNDRY_IS_PROPERTY_REACTION (self), NULL);

  return self->expression;
}

void
foundry_property_reaction_set_value (FoundryPropertyReaction *self,
                                     GtkExpression           *value)
{
  g_return_if_fail (FOUNDRY_IS_PROPERTY_REACTION (self));
  g_return_if_fail (!value || GTK_IS_EXPRESSION (value));

  if (value == self->expression)
    return;

  if (value)
    gtk_expression_ref (value);

  g_clear_pointer (&self->expression, gtk_expression_unref);
  self->expression = value;
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_VALUE]);
}

const char *
foundry_property_reaction_get_name (FoundryPropertyReaction *self)
{
  g_return_val_if_fail (FOUNDRY_IS_PROPERTY_REACTION (self), NULL);

  return self->name;
}

void
foundry_property_reaction_set_name (FoundryPropertyReaction *self,
                                    const char              *name)
{
  g_return_if_fail (FOUNDRY_IS_PROPERTY_REACTION (self));

  if (g_set_str (&self->name, name))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_NAME]);
}
