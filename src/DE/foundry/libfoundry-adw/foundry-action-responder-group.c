/* foundry-action-responder-group.c
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

#include <gtk/gtk.h>

#include "foundry-action-responder-group-private.h"

struct _FoundryActionResponderGroup
{
  GObject     parent_instance;
  GActionMap *map;
};

static inline GActionGroup *
get_wrapped (GActionGroup *group)
{
  return G_ACTION_GROUP (FOUNDRY_ACTION_RESPONDER_GROUP (group)->map);
}

#define WRAP(self, func, ...) \
  g_action_group_##func (get_wrapped(self), ##__VA_ARGS__)

static gboolean
foundry_action_responder_group_has_action (GActionGroup *group,
                                           const char   *action_name)
{
  return WRAP (group, has_action, action_name);
}

static char **
foundry_action_responder_group_list_actions (GActionGroup *group)
{
  return WRAP (group, list_actions);
}

static gboolean
foundry_action_responder_group_get_action_enabled (GActionGroup *group,
                                                   const char   *action_name)
{
  return WRAP (group, get_action_enabled, action_name);
}

static const GVariantType *
foundry_action_responder_group_get_action_parameter_type (GActionGroup *group,
                                                          const char   *action_name)
{
  return WRAP (group, get_action_parameter_type, action_name);
}

static const GVariantType *
foundry_action_responder_group_get_action_state_type (GActionGroup *group,
                                                      const char   *action_name)
{
  return WRAP (group, get_action_state_type, action_name);
}

static GVariant *
foundry_action_responder_group_get_action_state_hint (GActionGroup *group,
                                                      const char   *action_name)
{
  return WRAP (group, get_action_state_hint, action_name);
}

static GVariant *
foundry_action_responder_group_get_action_state (GActionGroup *group,
                                                 const char   *action_name)
{
  return WRAP (group, get_action_state, action_name);
}

static void
foundry_action_responder_group_change_action_state (GActionGroup *group,
                                                    const char   *action_name,
                                                    GVariant     *value)
{
  WRAP (group, change_action_state, action_name, value);
}

static void
foundry_action_responder_group_activate_action (GActionGroup *group,
                                                const char   *action_name,
                                                GVariant     *parameter)
{
  WRAP (group, activate_action, action_name, parameter);
}

static gboolean
foundry_action_responder_group_query_action (GActionGroup        *group,
                                             const char          *action_name,
                                             gboolean            *enabled,
                                             const GVariantType **parameter_type,
                                             const GVariantType **state_type,
                                             GVariant           **state_hint,
                                             GVariant           **state)
{
  return WRAP (group, query_action, action_name, enabled, parameter_type, state_type, state_hint, state);
}

static void
action_group_iface_init (GActionGroupInterface *iface)
{
  iface->has_action = foundry_action_responder_group_has_action;
  iface->list_actions = foundry_action_responder_group_list_actions;
  iface->get_action_enabled = foundry_action_responder_group_get_action_enabled;
  iface->get_action_parameter_type = foundry_action_responder_group_get_action_parameter_type;
  iface->get_action_state_type = foundry_action_responder_group_get_action_state_type;
  iface->get_action_state_hint = foundry_action_responder_group_get_action_state_hint;
  iface->get_action_state = foundry_action_responder_group_get_action_state;
  iface->change_action_state = foundry_action_responder_group_change_action_state;
  iface->activate_action = foundry_action_responder_group_activate_action;
  iface->query_action = foundry_action_responder_group_query_action;
}

static void
foundry_action_responder_group_add_child (GtkBuildable *buildable,
                                          GtkBuilder   *builder,
                                          GObject      *object,
                                          const char   *type)
{
  if (FOUNDRY_IS_ACTION_RESPONDER (object))
    foundry_action_responder_group_add (FOUNDRY_ACTION_RESPONDER_GROUP (buildable),
                                    FOUNDRY_ACTION_RESPONDER (object));
  else
    g_critical ("Cannot add child typed %s to %s",
                G_OBJECT_TYPE_NAME (object),
                G_OBJECT_TYPE_NAME (buildable));
}

static void
buildable_iface_init (GtkBuildableIface *iface)
{
  iface->add_child = foundry_action_responder_group_add_child;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryActionResponderGroup, foundry_action_responder_group, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_ACTION_GROUP, action_group_iface_init)
                               G_IMPLEMENT_INTERFACE (GTK_TYPE_BUILDABLE, buildable_iface_init))

static void
foundry_action_responder_group_finalize (GObject *object)
{
  FoundryActionResponderGroup *self = (FoundryActionResponderGroup *)object;

  g_clear_object (&self->map);

  G_OBJECT_CLASS (foundry_action_responder_group_parent_class)->finalize (object);
}

static void
foundry_action_responder_group_class_init (FoundryActionResponderGroupClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_action_responder_group_finalize;
}

static void
foundry_action_responder_group_init (FoundryActionResponderGroup *self)
{
  self->map = G_ACTION_MAP (g_simple_action_group_new ());

  g_signal_connect_object (self->map,
                           "action-added",
                           G_CALLBACK (g_action_group_action_added),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (self->map,
                           "action-enabled-changed",
                           G_CALLBACK (g_action_group_action_enabled_changed),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (self->map,
                           "action-removed",
                           G_CALLBACK (g_action_group_action_removed),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (self->map,
                           "action-state-changed",
                           G_CALLBACK (g_action_group_action_state_changed),
                           self,
                           G_CONNECT_SWAPPED);
}

FoundryActionResponderGroup *
foundry_action_responder_group_new (void)
{
  return g_object_new (FOUNDRY_TYPE_ACTION_RESPONDER_GROUP, NULL);
}

void
foundry_action_responder_group_add (FoundryActionResponderGroup *self,
                                    FoundryActionResponder      *responder)
{
  g_return_if_fail (FOUNDRY_IS_ACTION_RESPONDER_GROUP (self));
  g_return_if_fail (FOUNDRY_IS_ACTION_RESPONDER (responder));

  g_action_map_add_action (self->map, G_ACTION (responder));
}

void
foundry_action_responder_group_remove (FoundryActionResponderGroup *self,
                                       FoundryActionResponder      *responder)
{
  g_return_if_fail (FOUNDRY_IS_ACTION_RESPONDER_GROUP (self));
  g_return_if_fail (FOUNDRY_IS_ACTION_RESPONDER (responder));

  g_action_map_remove_action (self->map, g_action_get_name (G_ACTION (responder)));
}
