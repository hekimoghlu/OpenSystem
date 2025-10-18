/* foundry-action-muxer.c
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

#include "gsettings-mapping.h"
#include "foundry-action-muxer.h"

struct _FoundryActionMuxer
{
  GObject              parent_instance;
  GPtrArray           *action_groups;
  const FoundryAction *actions;
  GHashTable          *actions_disabled;
  GHashTable          *pspec_name_to_action;
  gpointer             instance;
  gulong               instance_notify_handler;
  guint                n_recurse;
};

typedef struct
{
  FoundryActionMuxer *backptr;
  char               *prefix;
  GActionGroup       *action_group;
  GSignalGroup       *action_group_signals;
} PrefixedActionGroup;

static void action_group_iface_init (GActionGroupInterface *iface);

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryActionMuxer, foundry_action_muxer, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_ACTION_GROUP, action_group_iface_init))

static GQuark mixin_quark;

static void
prefixed_action_group_finalize (gpointer data)
{
  PrefixedActionGroup *pag = data;

  g_assert (pag->backptr == NULL);

  g_clear_object (&pag->action_group_signals);
  g_clear_object (&pag->action_group);
  g_clear_pointer (&pag->prefix, g_free);
}

static void
prefixed_action_group_unref (PrefixedActionGroup *pag)
{
  g_rc_box_release_full (pag, prefixed_action_group_finalize);
}

static void
prefixed_action_group_drop (PrefixedActionGroup *pag)
{
  g_signal_group_set_target (pag->action_group_signals, NULL);
  pag->backptr = NULL;
  prefixed_action_group_unref (pag);
}

static PrefixedActionGroup *
prefixed_action_group_ref (PrefixedActionGroup *pag)
{
  return g_rc_box_acquire (pag);
}

static GVariant *
get_property_state (gpointer            instance,
                    GParamSpec         *pspec,
                    const GVariantType *state_type)
{
  GValue value = G_VALUE_INIT;
  GVariant *ret;

  g_assert (G_IS_OBJECT (instance));
  g_assert (pspec != NULL);
  g_assert (state_type != NULL);

  g_value_init (&value, pspec->value_type);
  g_object_get_property (instance, pspec->name, &value);
  ret = g_settings_set_mapping (&value, state_type, NULL);
  g_value_unset (&value);

  return g_variant_ref_sink (ret);
}

static void
foundry_action_muxer_dispose (GObject *object)
{
  FoundryActionMuxer *self = (FoundryActionMuxer *)object;

  if (self->instance != NULL)
    {
      g_clear_signal_handler (&self->instance_notify_handler, self->instance);
      g_clear_weak_pointer (&self->instance);
    }

  if (self->action_groups->len > 0)
    g_ptr_array_remove_range (self->action_groups, 0, self->action_groups->len);

  self->actions = NULL;
  g_clear_pointer (&self->actions_disabled, g_hash_table_unref);

  G_OBJECT_CLASS (foundry_action_muxer_parent_class)->dispose (object);
}

static void
foundry_action_muxer_finalize (GObject *object)
{
  FoundryActionMuxer *self = (FoundryActionMuxer *)object;

  g_clear_pointer (&self->action_groups, g_ptr_array_unref);

  G_OBJECT_CLASS (foundry_action_muxer_parent_class)->finalize (object);
}

static void
foundry_action_muxer_class_init (FoundryActionMuxerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_action_muxer_dispose;
  object_class->finalize = foundry_action_muxer_finalize;
}

static void
foundry_action_muxer_init (FoundryActionMuxer *self)
{
  self->action_groups = g_ptr_array_new_with_free_func ((GDestroyNotify)prefixed_action_group_drop);
  self->actions_disabled = g_hash_table_new (NULL, NULL);
}

FoundryActionMuxer *
foundry_action_muxer_new (void)
{
  return g_object_new (FOUNDRY_TYPE_ACTION_MUXER, NULL);
}

/**
 * foundry_action_muxer_list_groups:
 * @self: a #FoundryActionMuxer
 *
 * Gets a list of group names in the muxer.
 *
 * Returns: (transfer full) (array zero-terminated=1) (element-type utf8):
 *   an array containing the names of groups within the muxer
 */
char **
foundry_action_muxer_list_groups (FoundryActionMuxer *self)
{
  GArray *ar;

  g_return_val_if_fail (FOUNDRY_IS_ACTION_MUXER (self), NULL);

  ar = g_array_new (TRUE, FALSE, sizeof (char *));

  for (guint i = 0; i < self->action_groups->len; i++)
    {
      const PrefixedActionGroup *pag = g_ptr_array_index (self->action_groups, i);
      char *prefix = g_strdup (pag->prefix);

      g_assert (prefix != NULL);
      g_assert (g_str_has_suffix (prefix, "."));

      *strrchr (prefix, '.') = 0;

      g_array_append_val (ar, prefix);
    }

  return (char **)(gpointer)g_array_free (ar, FALSE);
}

static void
foundry_action_muxer_action_group_action_added_cb (GActionGroup        *action_group,
                                                   const char          *action_name,
                                                   PrefixedActionGroup *pag)
{
  g_autofree char *full_name = NULL;

  g_assert (G_IS_ACTION_GROUP (action_group));
  g_assert (action_name != NULL);
  g_assert (pag != NULL);
  g_assert (pag->backptr != NULL);
  g_assert (FOUNDRY_IS_ACTION_MUXER (pag->backptr));
  g_assert ((gpointer)pag->backptr != (gpointer)action_group);

  full_name = g_strconcat (pag->prefix, action_name, NULL);
  g_action_group_action_added (G_ACTION_GROUP (pag->backptr), full_name);
}

static void
foundry_action_muxer_action_group_action_removed_cb (GActionGroup        *action_group,
                                                     const char          *action_name,
                                                     PrefixedActionGroup *pag)
{
  g_autofree char *full_name = NULL;

  g_assert (G_IS_ACTION_GROUP (action_group));
  g_assert (action_name != NULL);
  g_assert (pag != NULL);
  g_assert (pag->backptr != NULL);
  g_assert (FOUNDRY_IS_ACTION_MUXER (pag->backptr));
  g_assert ((gpointer)pag->backptr != (gpointer)action_group);

  full_name = g_strconcat (pag->prefix, action_name, NULL);
  g_action_group_action_removed (G_ACTION_GROUP (pag->backptr), full_name);
}

static void
foundry_action_muxer_action_group_action_enabled_changed_cb (GActionGroup        *action_group,
                                                             const char          *action_name,
                                                             gboolean             enabled,
                                                             PrefixedActionGroup *pag)
{
  g_autofree char *full_name = NULL;

  g_assert (G_IS_ACTION_GROUP (action_group));
  g_assert (action_name != NULL);
  g_assert (pag != NULL);
  g_assert (pag->backptr != NULL);
  g_assert (FOUNDRY_IS_ACTION_MUXER (pag->backptr));

  full_name = g_strconcat (pag->prefix, action_name, NULL);
  g_action_group_action_enabled_changed (G_ACTION_GROUP (pag->backptr), full_name, enabled);
}

static void
foundry_action_muxer_action_group_action_state_changed_cb (GActionGroup        *action_group,
                                                           const char          *action_name,
                                                           GVariant            *value,
                                                           PrefixedActionGroup *pag)
{
  g_autofree char *full_name = NULL;

  g_assert (G_IS_ACTION_GROUP (action_group));
  g_assert (action_name != NULL);
  g_assert (pag != NULL);
  g_assert (pag->backptr != NULL);
  g_assert (FOUNDRY_IS_ACTION_MUXER (pag->backptr));

  full_name = g_strconcat (pag->prefix, action_name, NULL);
  g_action_group_action_state_changed (G_ACTION_GROUP (pag->backptr), full_name, value);
}

void
foundry_action_muxer_insert_action_group (FoundryActionMuxer *self,
                                          const char         *prefix,
                                          GActionGroup       *action_group)
{
  g_autofree char *prefix_dot = NULL;

  g_return_if_fail (FOUNDRY_IS_ACTION_MUXER (self));
  g_return_if_fail (self->n_recurse == 0);
  g_return_if_fail (prefix != NULL);
  g_return_if_fail (!action_group || G_IS_ACTION_GROUP (action_group));
  g_return_if_fail ((gpointer)action_group != (gpointer)self);

  /* Protect against recursion via signal emission. We don't want anything to
   * mess with our GArray while we are actively processing actions. To do so is
   * invalid API use.
   */
  self->n_recurse++;

  /* Precalculate with a dot suffix so we can simplify lookups */
  prefix_dot = g_strconcat (prefix, ".", NULL);

  /* Find our matching action group by prefix, and then notify it has been
   * removed from our known actions.
   */
  for (guint i = 0; i < self->action_groups->len; i++)
    {
      const PrefixedActionGroup *pag = g_ptr_array_index (self->action_groups, i);
      g_auto(GStrv) action_names = NULL;

      g_assert (pag->prefix != NULL);
      g_assert (G_IS_ACTION_GROUP (pag->action_group));

      if (g_strcmp0 (pag->prefix, prefix_dot) != 0)
        continue;

      /* Clear signal group first, since it has weak pointers */
      g_signal_group_set_target (pag->action_group_signals, NULL);

      /* Retrieve list of all the action names so we can drop references */
      action_names = g_action_group_list_actions (pag->action_group);

      /* Remove this entry from our knowledge, clear pag because it now
       * points to potentially invalid memory.
       */
      pag = NULL;
      g_ptr_array_remove_index_fast (self->action_groups, i);

      /* Notify any actiongroup listeners of removed actions */
      for (guint j = 0; action_names[j]; j++)
        {
          g_autofree char *action_name = g_strconcat (prefix_dot, action_names[j], NULL);
          g_action_group_action_removed (G_ACTION_GROUP (self), action_name);
        }

      break;
    }

  /* If we got a new action group to replace it, setup tracking of the
   * action group and then notify of all the current actions.
   */
  if (action_group != NULL)
    {
      g_auto(GStrv) action_names = g_action_group_list_actions (action_group);
      PrefixedActionGroup *new_pag = g_rc_box_new0 (PrefixedActionGroup);

      new_pag->backptr = self;
      new_pag->prefix = g_strdup (prefix_dot);
      new_pag->action_group = g_object_ref (action_group);
      new_pag->action_group_signals = g_signal_group_new (G_TYPE_ACTION_GROUP);
      g_ptr_array_add (self->action_groups, new_pag);

      g_signal_group_connect_data (new_pag->action_group_signals,
                                   "action-added",
                                   G_CALLBACK (foundry_action_muxer_action_group_action_added_cb),
                                   prefixed_action_group_ref (new_pag),
                                   (GClosureNotify)prefixed_action_group_unref,
                                   0);
      g_signal_group_connect_data (new_pag->action_group_signals,
                                   "action-removed",
                                   G_CALLBACK (foundry_action_muxer_action_group_action_removed_cb),
                                   prefixed_action_group_ref (new_pag),
                                   (GClosureNotify)prefixed_action_group_unref,
                                   0);
      g_signal_group_connect_data (new_pag->action_group_signals,
                                   "action-enabled-changed",
                                   G_CALLBACK (foundry_action_muxer_action_group_action_enabled_changed_cb),
                                   prefixed_action_group_ref (new_pag),
                                   (GClosureNotify)prefixed_action_group_unref,
                                   0);
      g_signal_group_connect_data (new_pag->action_group_signals,
                                   "action-state-changed",
                                   G_CALLBACK (foundry_action_muxer_action_group_action_state_changed_cb),
                                   prefixed_action_group_ref (new_pag),
                                   (GClosureNotify)prefixed_action_group_unref,
                                   0);

      g_signal_group_set_target (new_pag->action_group_signals, action_group);

      for (guint j = 0; action_names[j]; j++)
        {
          g_autofree char *action_name = g_strconcat (prefix_dot, action_names[j], NULL);
          g_action_group_action_added (G_ACTION_GROUP (self), action_name);
        }
    }

  self->n_recurse--;
}

void
foundry_action_muxer_remove_action_group (FoundryActionMuxer *self,
                                          const char         *prefix)
{
  g_return_if_fail (FOUNDRY_IS_ACTION_MUXER (self));
  g_return_if_fail (prefix != NULL);

  foundry_action_muxer_insert_action_group (self, prefix, NULL);
}

/**
 * foundry_action_muxer_get_action_group:
 * @self: a #FoundryActionMuxer
 * @prefix: the name of the inserted action group
 *
 * Locates the #GActionGroup inserted as @prefix.
 *
 * If no group was found matching @group, %NULL is returned.
 *
 * Returns: (transfer none) (nullable): a #GActionGroup matching @prefix if
 *   found, otherwise %NULL.
 */
GActionGroup *
foundry_action_muxer_get_action_group (FoundryActionMuxer *self,
                                       const char         *prefix)
{
  g_autofree char *prefix_dot = NULL;

  g_return_val_if_fail (FOUNDRY_IS_ACTION_MUXER (self), NULL);
  g_return_val_if_fail (prefix!= NULL, NULL);

  prefix_dot = g_strconcat (prefix, ".", NULL);

  for (guint i = 0; i < self->action_groups->len; i++)
    {
      const PrefixedActionGroup *pag = g_ptr_array_index (self->action_groups, i);

      if (g_strcmp0 (pag->prefix, prefix_dot) == 0)
        return pag->action_group;
    }

  return NULL;
}

static gboolean
foundry_action_muxer_has_action (GActionGroup *group,
                                 const char   *action_name)
{
  FoundryActionMuxer *self = FOUNDRY_ACTION_MUXER (group);

  for (const FoundryAction *iter = self->actions; iter; iter = iter->next)
    {
      if (g_strcmp0 (iter->name, action_name) == 0)
        return TRUE;
    }

  for (guint i = 0; i < self->action_groups->len; i++)
    {
      const PrefixedActionGroup *pag = g_ptr_array_index (self->action_groups, i);

      if (g_str_has_prefix (action_name, pag->prefix))
        {
          const char *short_name = action_name + strlen (pag->prefix);

          if (g_action_group_has_action (pag->action_group, short_name))
            return TRUE;
        }
    }

  return FALSE;
}

static char **
foundry_action_muxer_list_actions (GActionGroup *group)
{
  FoundryActionMuxer *self = FOUNDRY_ACTION_MUXER (group);
  GArray *ar = g_array_new (TRUE, FALSE, sizeof (char *));

  for (const FoundryAction *iter = self->actions; iter; iter = iter->next)
    {
      char *name = g_strdup (iter->name);
      g_array_append_val (ar, name);
    }

  for (guint i = 0; i < self->action_groups->len; i++)
    {
      const PrefixedActionGroup *pag = g_ptr_array_index (self->action_groups, i);
      g_auto(GStrv) action_names = g_action_group_list_actions (pag->action_group);

      for (guint j = 0; action_names[j]; j++)
        {
          char *full_action_name = g_strconcat (pag->prefix, action_names[j], NULL);
          g_array_append_val (ar, full_action_name);
        }
    }

  return (char **)(gpointer)g_array_free (ar, FALSE);
}

static gboolean
foundry_action_muxer_get_action_enabled (GActionGroup *group,
                                         const char   *action_name)
{
  FoundryActionMuxer *self = FOUNDRY_ACTION_MUXER (group);

  for (const FoundryAction *iter = self->actions; iter; iter = iter->next)
    {
      if (g_strcmp0 (action_name, iter->name) == 0)
        return !g_hash_table_contains (self->actions_disabled, GUINT_TO_POINTER (iter->position));
    }

  for (guint i = 0; i < self->action_groups->len; i++)
    {
      const PrefixedActionGroup *pag = g_ptr_array_index (self->action_groups, i);

      if (g_str_has_prefix (action_name, pag->prefix))
        {
          const char *short_name = action_name + strlen (pag->prefix);

          if (g_action_group_has_action (pag->action_group, short_name))
            return g_action_group_get_action_enabled (pag->action_group, short_name);
        }
    }

  return FALSE;
}

static GVariant *
foundry_action_muxer_get_action_state (GActionGroup *group,
                                       const char   *action_name)
{
  FoundryActionMuxer *self = FOUNDRY_ACTION_MUXER (group);

  for (const FoundryAction *iter = self->actions; iter; iter = iter->next)
    {
      if (g_strcmp0 (iter->name, action_name) == 0)
        {
          if (iter->pspec != NULL && self->instance != NULL)
            return get_property_state (self->instance, iter->pspec, iter->state_type);
          return NULL;
        }
    }

  for (guint i = 0; i < self->action_groups->len; i++)
    {
      const PrefixedActionGroup *pag = g_ptr_array_index (self->action_groups, i);

      if (g_str_has_prefix (action_name, pag->prefix))
        {
          const char *short_name = action_name + strlen (pag->prefix);

          if (g_action_group_has_action (pag->action_group, short_name))
            return g_action_group_get_action_state (pag->action_group, short_name);
        }
    }

  return NULL;
}

static GVariant *
foundry_action_muxer_get_action_state_hint (GActionGroup *group,
                                            const char   *action_name)
{
  FoundryActionMuxer *self = FOUNDRY_ACTION_MUXER (group);

  for (const FoundryAction *iter = self->actions; iter; iter = iter->next)
    {
      if (g_strcmp0 (iter->name, action_name) == 0)
        {
          if (iter->pspec != NULL)
            {
              if (iter->pspec->value_type == G_TYPE_INT)
                {
                  GParamSpecInt *pspec = (GParamSpecInt *)iter->pspec;
                  return g_variant_new ("(ii)", pspec->minimum, pspec->maximum);
                }
              else if (iter->pspec->value_type == G_TYPE_UINT)
                {
                  GParamSpecUInt *pspec = (GParamSpecUInt *)iter->pspec;
                  return g_variant_new ("(uu)", pspec->minimum, pspec->maximum);
                }
              else if (iter->pspec->value_type == G_TYPE_FLOAT)
                {
                  GParamSpecFloat *pspec = (GParamSpecFloat *)iter->pspec;
                  return g_variant_new ("(dd)", (double)pspec->minimum, (double)pspec->maximum);
                }
              else if (iter->pspec->value_type == G_TYPE_DOUBLE)
                {
                  GParamSpecDouble *pspec = (GParamSpecDouble *)iter->pspec;
                  return g_variant_new ("(dd)", pspec->minimum, pspec->maximum);
                }
            }

          return NULL;
        }
    }

  for (guint i = 0; i < self->action_groups->len; i++)
    {
      const PrefixedActionGroup *pag = g_ptr_array_index (self->action_groups, i);

      if (g_str_has_prefix (action_name, pag->prefix))
        {
          const char *short_name = action_name + strlen (pag->prefix);

          if (g_action_group_has_action (pag->action_group, short_name))
            return g_action_group_get_action_state_hint (pag->action_group, short_name);
        }
    }

  return NULL;
}

static void
foundry_action_muxer_change_action_state (GActionGroup *group,
                                          const char   *action_name,
                                          GVariant     *value)
{
  FoundryActionMuxer *self = FOUNDRY_ACTION_MUXER (group);

  for (const FoundryAction *iter = self->actions; iter; iter = iter->next)
    {
      if (g_strcmp0 (iter->name, action_name) == 0)
        {
          if (iter->pspec != NULL && self->instance != NULL)
            {
              GValue gvalue = G_VALUE_INIT;
              g_value_init (&gvalue, iter->pspec->value_type);
              g_settings_get_mapping (&gvalue, value, NULL);
              g_object_set_property (self->instance, iter->pspec->name, &gvalue);
              g_value_unset (&gvalue);
            }

          return;
        }
    }

  for (guint i = 0; i < self->action_groups->len; i++)
    {
      const PrefixedActionGroup *pag = g_ptr_array_index (self->action_groups, i);

      if (g_str_has_prefix (action_name, pag->prefix))
        {
          const char *short_name = action_name + strlen (pag->prefix);

          if (g_action_group_has_action (pag->action_group, short_name))
            {
              g_action_group_change_action_state (pag->action_group, short_name, value);
              break;
            }
        }
    }
}

static const GVariantType *
foundry_action_muxer_get_action_state_type (GActionGroup *group,
                                            const char   *action_name)
{
  FoundryActionMuxer *self = FOUNDRY_ACTION_MUXER (group);

  for (const FoundryAction *iter = self->actions; iter; iter = iter->next)
    {
      if (g_strcmp0 (iter->name, action_name) == 0)
        return iter->state_type;
    }

  for (guint i = 0; i < self->action_groups->len; i++)
    {
      const PrefixedActionGroup *pag = g_ptr_array_index (self->action_groups, i);

      if (g_str_has_prefix (action_name, pag->prefix))
        {
          const char *short_name = action_name + strlen (pag->prefix);

          if (g_action_group_has_action (pag->action_group, short_name))
            return g_action_group_get_action_state_type (pag->action_group, short_name);
        }
    }

  return NULL;
}

static void
foundry_action_muxer_activate_action (GActionGroup *group,
                                      const char   *action_name,
                                      GVariant     *parameter)
{
  FoundryActionMuxer *self = FOUNDRY_ACTION_MUXER (group);

  for (const FoundryAction *iter = self->actions; iter; iter = iter->next)
    {
      if (g_strcmp0 (iter->name, action_name) == 0)
        {
          if (iter->pspec != NULL)
            {
              if (iter->pspec->value_type == G_TYPE_BOOLEAN)
                {
                  gboolean value;

                  g_return_if_fail (parameter == NULL);

                  g_object_get (self->instance, iter->pspec->name, &value, NULL);
                  value = !value;
                  g_object_set (self->instance, iter->pspec->name, value, NULL);
                }
              else
                {
                  g_return_if_fail (parameter != NULL && g_variant_is_of_type (parameter, iter->state_type));

                  foundry_action_muxer_change_action_state (group, action_name, parameter);
                }

            }
          else
            {
              iter->activate (self->instance, iter->name, parameter);
            }

          return;
        }
    }

  for (guint i = 0; i < self->action_groups->len; i++)
    {
      const PrefixedActionGroup *pag = g_ptr_array_index (self->action_groups, i);

      if (g_str_has_prefix (action_name, pag->prefix))
        {
          const char *short_name = action_name + strlen (pag->prefix);

          if (g_action_group_has_action (pag->action_group, short_name))
            {
              g_action_group_activate_action (pag->action_group, short_name, parameter);
              break;
            }
        }
    }
}

static const GVariantType *
foundry_action_muxer_get_action_parameter_type (GActionGroup *group,
                                                const char   *action_name)
{
  FoundryActionMuxer *self = FOUNDRY_ACTION_MUXER (group);

  for (const FoundryAction *iter = self->actions; iter; iter = iter->next)
    {
      if (g_strcmp0 (iter->name, action_name) == 0)
        return iter->parameter_type;
    }

  for (guint i = 0; i < self->action_groups->len; i++)
    {
      const PrefixedActionGroup *pag = g_ptr_array_index (self->action_groups, i);

      if (g_str_has_prefix (action_name, pag->prefix))
        {
          const char *short_name = action_name + strlen (pag->prefix);

          if (g_action_group_has_action (pag->action_group, short_name))
            return g_action_group_get_action_parameter_type (pag->action_group, short_name);
        }
    }

  return NULL;
}

static void
action_group_iface_init (GActionGroupInterface *iface)
{
  iface->has_action = foundry_action_muxer_has_action;
  iface->list_actions = foundry_action_muxer_list_actions;
  iface->get_action_parameter_type = foundry_action_muxer_get_action_parameter_type;
  iface->get_action_enabled = foundry_action_muxer_get_action_enabled;
  iface->get_action_state = foundry_action_muxer_get_action_state;
  iface->get_action_state_hint = foundry_action_muxer_get_action_state_hint;
  iface->get_action_state_type = foundry_action_muxer_get_action_state_type;
  iface->change_action_state = foundry_action_muxer_change_action_state;
  iface->activate_action = foundry_action_muxer_activate_action;
}

void
foundry_action_muxer_remove_all (FoundryActionMuxer *self)
{
  g_auto(GStrv) action_groups = NULL;

  g_return_if_fail (FOUNDRY_IS_ACTION_MUXER (self));

  if ((action_groups = foundry_action_muxer_list_actions (G_ACTION_GROUP (self))))
    {
      for (guint i = 0; action_groups[i]; i++)
        foundry_action_muxer_remove_action_group (self, action_groups[i]);
    }
}

gboolean
foundry_action_muxer_get_enabled (FoundryActionMuxer  *self,
                                  const FoundryAction *action)
{
  g_return_val_if_fail (FOUNDRY_IS_ACTION_MUXER (self), FALSE);
  g_return_val_if_fail (action != NULL, FALSE);

  return !g_hash_table_contains (self->actions_disabled, GUINT_TO_POINTER (action->position));
}

void
foundry_action_muxer_set_enabled (FoundryActionMuxer  *self,
                                  const FoundryAction *action,
                                  gboolean             enabled)
{
  gboolean disabled = !enabled;

  g_return_if_fail (FOUNDRY_IS_ACTION_MUXER (self));
  g_return_if_fail (action != NULL);

  if (disabled != g_hash_table_contains (self->actions_disabled, GUINT_TO_POINTER (action->position)))
    {
      if (disabled)
        g_hash_table_add (self->actions_disabled, GUINT_TO_POINTER (action->position));
      else
        g_hash_table_remove (self->actions_disabled, GUINT_TO_POINTER (action->position));

      g_action_group_action_enabled_changed (G_ACTION_GROUP (self), action->name, !disabled);
    }
}

static void
foundry_action_muxer_property_action_notify_cb (FoundryActionMuxer *self,
                                                GParamSpec         *pspec,
                                                gpointer            instance)
{
  g_autoptr(GVariant) state = NULL;
  const FoundryAction *action;

  g_assert (FOUNDRY_IS_ACTION_MUXER (self));
  g_assert (pspec != NULL);
  g_assert (G_IS_OBJECT (instance));

  if (!(action = g_hash_table_lookup (self->pspec_name_to_action, pspec->name)))
    return;

  state = get_property_state (instance, action->pspec, action->state_type);

  g_action_group_action_state_changed (G_ACTION_GROUP (self), action->name, state);
}

static void
foundry_action_muxer_add_property_action (FoundryActionMuxer  *self,
                                          gpointer             instance,
                                          const FoundryAction *action)
{
  g_assert (FOUNDRY_IS_ACTION_MUXER (self));
  g_assert (G_IS_OBJECT (instance));
  g_assert (action != NULL);
  g_assert (action->pspec != NULL);
  g_assert (g_type_is_a (G_OBJECT_TYPE (instance), action->owner));

  if (self->pspec_name_to_action == NULL)
    self->pspec_name_to_action = g_hash_table_new (NULL, NULL);

  g_hash_table_insert (self->pspec_name_to_action,
                       (gpointer)action->pspec->name,
                       (gpointer)action);

  if (self->instance_notify_handler == 0)
    self->instance_notify_handler =
      g_signal_connect_object (instance,
                               "notify",
                               G_CALLBACK (foundry_action_muxer_property_action_notify_cb),
                               self,
                               G_CONNECT_SWAPPED);

  g_action_group_action_added (G_ACTION_GROUP (self), action->name);
}

static void
foundry_action_muxer_add_action (FoundryActionMuxer  *self,
                                 gpointer             instance,
                                 const FoundryAction *action)
{
  g_assert (FOUNDRY_IS_ACTION_MUXER (self));
  g_assert (G_IS_OBJECT (instance));
  g_assert (action != NULL);
  g_assert (g_type_is_a (G_OBJECT_TYPE (instance), action->owner));

  g_action_group_action_added (G_ACTION_GROUP (self), action->name);
}

void
foundry_action_muxer_connect_actions (FoundryActionMuxer  *self,
                                      gpointer             instance,
                                      const FoundryAction *actions)
{
  g_return_if_fail (FOUNDRY_IS_ACTION_MUXER (self));
  g_return_if_fail (G_IS_OBJECT (instance));
  g_return_if_fail (self->instance == NULL);

  if (actions == NULL)
    return;

  g_set_weak_pointer (&self->instance, instance);

  self->actions = actions;

  for (const FoundryAction *iter = actions; iter; iter = iter->next)
    {
      g_assert (iter->next == NULL ||
                iter->position == iter->next->position + 1);
      g_assert (iter->pspec != NULL || iter->activate != NULL);

      if (iter->pspec != NULL)
        foundry_action_muxer_add_property_action (self, instance, iter);
      else
        foundry_action_muxer_add_action (self, instance, iter);
    }
}

void
foundry_action_mixin_init (FoundryActionMixin *mixin,
                           GObjectClass       *object_class)
{
  g_return_if_fail (mixin != NULL);
  g_return_if_fail (G_IS_OBJECT_CLASS (object_class));

  if (!mixin_quark)
    mixin_quark = g_quark_from_static_string ("foundry-action-mixin");

  mixin->object_class = object_class;
}

/**
 * foundry_action_mixin_install_action: (skip)
 * @mixin: an `FoundryActionMixin`
 * @action_name: a prefixed action name, such as "clipboard.paste"
 * @parameter_type: (nullable): the parameter type
 * @activate: (scope notified): callback to use when the action is activated
 *
 * This should be called at class initialization time to specify
 * actions to be added for all instances of this class.
 *
 * Actions installed by this function are stateless. The only state
 * they have is whether they are enabled or not.
 */
void
foundry_action_mixin_install_action (FoundryActionMixin        *mixin,
                                     const char                *action_name,
                                     const char                *parameter_type,
                                     FoundryActionActivateFunc  activate)
{
  FoundryAction *action;

  g_return_if_fail (mixin != NULL);
  g_return_if_fail (G_IS_OBJECT_CLASS (mixin->object_class));

  action = g_new0 (FoundryAction, 1);
  action->owner = G_OBJECT_CLASS_TYPE (mixin->object_class);
  action->name = g_intern_string (action_name);
  if (parameter_type != NULL)
    action->parameter_type = g_variant_type_new (parameter_type);
  action->activate = (FoundryActionActivateFunc)activate;
  action->position = ++mixin->n_actions;
  action->next = mixin->actions;
  mixin->actions = action;
}

static const GVariantType *
determine_type (GParamSpec *pspec)
{
  if (G_TYPE_IS_ENUM (pspec->value_type))
    return G_VARIANT_TYPE_STRING;

  switch (pspec->value_type)
    {
    case G_TYPE_BOOLEAN:
      return G_VARIANT_TYPE_BOOLEAN;

    case G_TYPE_INT:
      return G_VARIANT_TYPE_INT32;

    case G_TYPE_UINT:
      return G_VARIANT_TYPE_UINT32;

    case G_TYPE_DOUBLE:
    case G_TYPE_FLOAT:
      return G_VARIANT_TYPE_DOUBLE;

    case G_TYPE_STRING:
      return G_VARIANT_TYPE_STRING;

    default:
      g_critical ("Unable to use foundry_action_mixin_install_property_action with property '%s:%s' of type '%s'",
                  g_type_name (pspec->owner_type), pspec->name, g_type_name (pspec->value_type));
      return NULL;
    }
}

/**
 * foundry_action_mixin_install_property_action: (skip)
 * @mixin: an `FoundryActionMixin`
 * @action_name: name of the action
 * @property_name: name of the property in instances of @mixin
 *   or any parent class.
 *
 * Installs an action called @action_name on @mmixin and
 * binds its state to the value of the @property_name property.
 *
 * This function will perform a few santity checks on the property selected
 * via @property_name. Namely, the property must exist, must be readable,
 * writable and must not be construct-only. There are also restrictions
 * on the type of the given property, it must be boolean, int, unsigned int,
 * double or string. If any of these conditions are not met, a critical
 * warning will be printed and no action will be added.
 *
 * The state type of the action matches the property type.
 *
 * If the property is boolean, the action will have no parameter and
 * toggle the property value. Otherwise, the action will have a parameter
 * of the same type as the property.
 */
void
foundry_action_mixin_install_property_action (FoundryActionMixin *mixin,
                                              const char         *action_name,
                                              const char         *property_name)
{
  const GVariantType *state_type;
  FoundryAction *action;
  GParamSpec *pspec;

  g_return_if_fail (mixin != NULL);
  g_return_if_fail (action_name != NULL);
  g_return_if_fail (property_name != NULL);
  g_return_if_fail (G_IS_OBJECT_CLASS (mixin->object_class));

  if (!(pspec = g_object_class_find_property (mixin->object_class, property_name)))
    {
      g_critical ("Attempted to use non-existent property '%s:%s' for foundry_action_mixin_install_property_action",
                  G_OBJECT_CLASS_NAME (mixin->object_class), property_name);
      return;
    }

  if (~pspec->flags & G_PARAM_READABLE || ~pspec->flags & G_PARAM_WRITABLE || pspec->flags & G_PARAM_CONSTRUCT_ONLY)
    {
      g_critical ("Property '%s:%s' used with foundry_action_mixin_install_property_action must be readable, writable, and not construct-only",
                  G_OBJECT_CLASS_NAME (mixin->object_class), property_name);
      return;
    }

  state_type = determine_type (pspec);

  if (!state_type)
    return;

  action = g_new0 (FoundryAction, 1);
  action->owner = G_TYPE_FROM_CLASS (mixin->object_class);
  action->name = g_intern_string (action_name);
  action->pspec = pspec;
  action->state_type = state_type;
  if (action->pspec->value_type != G_TYPE_BOOLEAN)
    action->parameter_type = action->state_type;
  action->position = ++mixin->n_actions;
  action->next = mixin->actions;

  mixin->actions = action;
}

/**
 * foundry_action_mixin_get_action_muxer: (skip)
 * @instance: a #FoundryActionMuxer
 *
 * Returns: (transfer none) (nullable): an #FoundryActionMuxer or %NULL
 */
FoundryActionMuxer *
foundry_action_mixin_get_action_muxer (gpointer instance)
{
  return g_object_get_qdata (instance, mixin_quark);
}

gboolean
foundry_action_mixin_get_enabled (gpointer    instance,
                                  const char *action_name)
{
  FoundryActionMuxer *muxer;

  g_return_val_if_fail (G_IS_OBJECT (instance), FALSE);
  g_return_val_if_fail (action_name != NULL, FALSE);

  muxer = foundry_action_mixin_get_action_muxer (instance);

  for (const FoundryAction *iter = muxer->actions; iter; iter = iter->next)
    {
      if (g_strcmp0 (iter->name, action_name) == 0)
        return foundry_action_muxer_get_enabled (muxer, iter);
    }

  return FALSE;
}

void
foundry_action_mixin_set_enabled (gpointer    instance,
                                  const char *action_name,
                                  gboolean    enabled)
{
  FoundryActionMuxer *muxer;

  g_return_if_fail (G_IS_OBJECT (instance));
  g_return_if_fail (action_name != NULL);

  muxer = foundry_action_mixin_get_action_muxer (instance);

  for (const FoundryAction *iter = muxer->actions; iter; iter = iter->next)
    {
      if (g_strcmp0 (iter->name, action_name) == 0)
        {
          foundry_action_muxer_set_enabled (muxer, iter, enabled);
          break;
        }
    }
}

void
foundry_action_mixin_constructed (const FoundryActionMixin *mixin,
                                  gpointer                  instance)
{
  FoundryActionMuxer *muxer;

  g_return_if_fail (mixin != NULL);
  g_return_if_fail (G_IS_OBJECT (instance));

  muxer = foundry_action_muxer_new ();
  g_object_set_qdata_full (instance, mixin_quark, muxer, g_object_unref);
  foundry_action_muxer_connect_actions (muxer, instance, mixin->actions);
}
