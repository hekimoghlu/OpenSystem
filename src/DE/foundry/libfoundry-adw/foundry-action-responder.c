/* foundry-action-responder.c
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

#include <gio/gio.h>

#include "foundry-action-responder-private.h"

struct _FoundryActionResponder
{
  FoundryResponder  parent_instance;
  GVariantType     *parameter_type;
  GVariantType     *state_type;
  char             *name;
  GVariant         *state;
  guint             enabled : 1;
};

enum {
  PROP_0,
  PROP_ENABLED,
  PROP_NAME,
  PROP_PARAMETER_TYPE,
  PROP_STATE_TYPE,
  PROP_STATE,
  N_PROPS
};

static const char *
action_iface_get_name (GAction *action)
{
  return FOUNDRY_ACTION_RESPONDER (action)->name;
}

static const GVariantType *
action_iface_get_parameter_type (GAction *action)
{
  return FOUNDRY_ACTION_RESPONDER (action)->parameter_type;
}

static const GVariantType *
action_iface_get_state_type (GAction *action)
{
  return FOUNDRY_ACTION_RESPONDER (action)->state_type;
}

static GVariant *
action_iface_get_state_hint (GAction *action)
{
  return NULL;
}

static GVariant *
action_iface_get_state (GAction *action)
{
  return FOUNDRY_ACTION_RESPONDER (action)->state;
}

static void
action_iface_change_state (GAction  *action,
                           GVariant *value)
{
  foundry_action_responder_set_state (FOUNDRY_ACTION_RESPONDER (action), value);
}

static void
action_iface_activate (GAction  *action,
                       GVariant *parameter)
{
  foundry_reaction_react (FOUNDRY_REACTION (action));
}

static gboolean
action_iface_get_enabled (GAction *action)
{
  return FOUNDRY_ACTION_RESPONDER (action)->enabled;
}

static void
action_iface_init (GActionInterface *iface)
{
  iface->get_name = action_iface_get_name;
  iface->get_parameter_type = action_iface_get_parameter_type;
  iface->get_state_type = action_iface_get_state_type;
  iface->get_state_hint = action_iface_get_state_hint;
  iface->get_state = action_iface_get_state;
  iface->change_state = action_iface_change_state;
  iface->activate = action_iface_activate;
  iface->get_enabled = action_iface_get_enabled;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryActionResponder, foundry_action_responder, FOUNDRY_TYPE_RESPONDER,
                               G_IMPLEMENT_INTERFACE (G_TYPE_ACTION, action_iface_init))

static GParamSpec *properties[N_PROPS];

static void
foundry_action_responder_finalize (GObject *object)
{
  FoundryActionResponder *self = (FoundryActionResponder *)object;

  g_clear_pointer (&self->name, g_free);
  g_clear_pointer (&self->state, g_variant_unref);
  g_clear_pointer (&self->parameter_type, g_variant_type_free);
  g_clear_pointer (&self->state_type, g_variant_type_free);

  G_OBJECT_CLASS (foundry_action_responder_parent_class)->finalize (object);
}

static void
foundry_action_responder_get_property (GObject    *object,
                                   guint       prop_id,
                                   GValue     *value,
                                   GParamSpec *pspec)
{
  FoundryActionResponder *self = FOUNDRY_ACTION_RESPONDER (object);

  switch (prop_id)
    {
    case PROP_NAME:
      g_value_set_string (value, self->name);
      break;

    case PROP_ENABLED:
      g_value_set_boolean (value, foundry_action_responder_get_enabled (self));
      break;

    case PROP_PARAMETER_TYPE:
      g_value_set_boxed (value, self->parameter_type);
      break;

    case PROP_STATE_TYPE:
      g_value_set_boxed (value, self->state_type);
      break;

    case PROP_STATE:
      g_value_set_variant (value, self->state);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_action_responder_set_property (GObject      *object,
                                   guint         prop_id,
                                   const GValue *value,
                                   GParamSpec   *pspec)
{
  FoundryActionResponder *self = FOUNDRY_ACTION_RESPONDER (object);

  switch (prop_id)
    {
    case PROP_NAME:
      self->name = g_value_dup_string (value);
      break;

    case PROP_ENABLED:
      foundry_action_responder_set_enabled (self, g_value_get_boolean (value));
      break;

    case PROP_PARAMETER_TYPE:
      self->parameter_type = g_value_dup_boxed (value);
      break;

    case PROP_STATE_TYPE:
      self->state_type = g_value_dup_boxed (value);
      break;

    case PROP_STATE:
      foundry_action_responder_set_state (self, g_value_get_boxed (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_action_responder_class_init (FoundryActionResponderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_action_responder_finalize;
  object_class->get_property = foundry_action_responder_get_property;
  object_class->set_property = foundry_action_responder_set_property;

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ENABLED] =
    g_param_spec_boolean ("enabled", NULL, NULL,
                          TRUE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_PARAMETER_TYPE] =
    g_param_spec_boxed ("parameter-type", NULL, NULL,
                         G_TYPE_VARIANT_TYPE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_STATE_TYPE] =
    g_param_spec_boxed ("state-type", NULL, NULL,
                         G_TYPE_VARIANT_TYPE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_STATE] =
    g_param_spec_variant ("state", NULL, NULL,
                          G_VARIANT_TYPE_ANY,
                          NULL,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_action_responder_init (FoundryActionResponder *self)
{
  self->enabled = TRUE;
}

gboolean
foundry_action_responder_get_enabled (FoundryActionResponder *self)
{
  g_return_val_if_fail (FOUNDRY_IS_ACTION_RESPONDER (self), FALSE);

  return self->enabled;
}

void
foundry_action_responder_set_enabled (FoundryActionResponder *self,
                                      gboolean                enabled)
{
  g_return_if_fail (FOUNDRY_IS_ACTION_RESPONDER (self));

  enabled = !!enabled;

  if (enabled == self->enabled)
    return;

  self->enabled = enabled;
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ENABLED]);
}

/**
 * foundry_action_responder_get_state:
 * @self: a [class@Foundry.ActionResponder]
 *
 * Returns: (transfer none) (nullable):
 */
GVariant *
foundry_action_responder_get_state (FoundryActionResponder *self)
{
  g_return_val_if_fail (FOUNDRY_IS_ACTION_RESPONDER (self), NULL);

  return self->state;
}

void
foundry_action_responder_set_state (FoundryActionResponder *self,
                                    GVariant               *state)
{
  g_return_if_fail (FOUNDRY_IS_ACTION_RESPONDER (self));

  if (self->state == state ||
      (self->state && state && g_variant_equal (self->state, state)))
    return;

  if (state != NULL)
    g_variant_ref_sink (state);
  g_clear_pointer (&self->state, g_variant_unref);
  self->state = state;

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_STATE]);
}
