/* foundry-input-combo.c
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

#include "foundry-input-choice.h"
#include "foundry-input-combo.h"
#include "foundry-input-validator.h"
#include "foundry-util-private.h"

struct _FoundryInputCombo
{
  FoundryInput  parent_instance;
  GMutex        mutex;
  GListModel   *choices;
};

enum {
  PROP_0,
  PROP_CHOICE,
  PROP_CHOICES,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryInputCombo, foundry_input_combo, FOUNDRY_TYPE_INPUT)

static GParamSpec *properties[N_PROPS];

static void
foundry_input_combo_finalize (GObject *object)
{
  FoundryInputCombo *self = (FoundryInputCombo *)object;

  g_clear_object (&self->choices);
  g_mutex_clear (&self->mutex);

  G_OBJECT_CLASS (foundry_input_combo_parent_class)->finalize (object);
}

static void
foundry_input_combo_get_property (GObject    *object,
                                  guint       prop_id,
                                  GValue     *value,
                                  GParamSpec *pspec)
{
  FoundryInputCombo *self = FOUNDRY_INPUT_COMBO (object);

  switch (prop_id)
    {
    case PROP_CHOICE:
      g_value_take_object (value, foundry_input_combo_dup_choice (self));
      break;

    case PROP_CHOICES:
      g_value_take_object (value, foundry_input_combo_list_choices (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_combo_set_property (GObject      *object,
                                  guint         prop_id,
                                  const GValue *value,
                                  GParamSpec   *pspec)
{
  FoundryInputCombo *self = FOUNDRY_INPUT_COMBO (object);

  switch (prop_id)
    {
    case PROP_CHOICE:
      foundry_input_combo_set_choice (self, g_value_get_object (value));
      break;

    case PROP_CHOICES:
      self->choices = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_combo_class_init (FoundryInputComboClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_input_combo_finalize;
  object_class->get_property = foundry_input_combo_get_property;
  object_class->set_property = foundry_input_combo_set_property;

  properties[PROP_CHOICE] =
    g_param_spec_object ("choice", NULL, NULL,
                         FOUNDRY_TYPE_INPUT_CHOICE,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_CHOICES] =
    g_param_spec_object ("choices", NULL, NULL,
                         G_TYPE_LIST_MODEL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_input_combo_init (FoundryInputCombo *self)
{
  g_mutex_init (&self->mutex);
}

/**
 * foundry_input_combo_list_choices:
 * @self: a [class@Foundry.InputCombo]
 *
 * Returns: (transfer full) (nullable): a [iface@Gio.ListModel] of
 *   [class@Foundry.InputChoice].
 */
GListModel *
foundry_input_combo_list_choices (FoundryInputCombo *self)
{
  GListModel *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_INPUT_COMBO (self), NULL);

  g_mutex_lock (&self->mutex);
  g_set_object (&ret, self->choices);
  g_mutex_unlock (&self->mutex);

  return g_steal_pointer (&ret);
}

/**
 * foundry_input_combo_dup_choice:
 * @self: a [class@Foundry.InputCombo]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryInputChoice *
foundry_input_combo_dup_choice (FoundryInputCombo *self)
{
  FoundryInputChoice *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_INPUT_COMBO (self), NULL);

  g_mutex_lock (&self->mutex);

  if (self->choices != NULL)
    {
      guint n_items = g_list_model_get_n_items (self->choices);

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryInputChoice) choice = g_list_model_get_item (self->choices, i);

          if (foundry_input_choice_get_selected (choice))
            {
              ret = g_steal_pointer (&choice);
              break;
            }
        }
    }

  g_mutex_unlock (&self->mutex);

  return g_steal_pointer (&ret);
}

void
foundry_input_combo_set_choice (FoundryInputCombo  *self,
                                FoundryInputChoice *choice)
{
  g_return_if_fail (FOUNDRY_IS_INPUT_COMBO (self));

  if (foundry_input_choice_get_selected (choice))
    return;

  g_mutex_lock (&self->mutex);

  if (self->choices != NULL)
    {
      guint n_items = g_list_model_get_n_items (self->choices);

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryInputChoice) item = g_list_model_get_item (self->choices, i);

          foundry_input_choice_set_selected (item, FALSE);
        }
    }

  g_mutex_unlock (&self->mutex);

  foundry_input_choice_set_selected (choice, TRUE);

  foundry_notify_pspec_in_main (G_OBJECT (self), properties[PROP_CHOICE]);
}

/**
 * foundry_input_combo_new:
 * @validator: (transfer full) (nullable): optional validator
 *
 * Returns: (transfer full):
 */
FoundryInput *
foundry_input_combo_new (const char            *title,
                         const char            *subtitle,
                         FoundryInputValidator *validator,
                         GListModel            *choices)
{
  g_autoptr(FoundryInputValidator) stolen = NULL;

  g_return_val_if_fail (!validator || FOUNDRY_IS_INPUT_VALIDATOR (validator), NULL);
  g_return_val_if_fail (G_IS_LIST_MODEL (choices), NULL);

  stolen = validator;

  return g_object_new (FOUNDRY_TYPE_INPUT_COMBO,
                       "title", title,
                       "subtitle", subtitle,
                       "validator", validator,
                       "choices", choices,
                       NULL);
}
