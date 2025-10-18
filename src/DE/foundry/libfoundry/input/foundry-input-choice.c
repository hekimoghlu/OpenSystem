/* foundry-input-choice.c
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
#include "foundry-util-private.h"

struct _FoundryInputChoice
{
  FoundryInput parent_instance;
  GMutex mutex;
  GObject *item;
  guint selected : 1;
};

enum {
  PROP_0,
  PROP_ITEM,
  PROP_SELECTED,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryInputChoice, foundry_input_choice, FOUNDRY_TYPE_INPUT)

static GParamSpec *properties[N_PROPS];

static void
foundry_input_choice_finalize (GObject *object)
{
  FoundryInputChoice *self = (FoundryInputChoice *)object;

  g_clear_object (&self->item);
  g_mutex_clear (&self->mutex);

  G_OBJECT_CLASS (foundry_input_choice_parent_class)->finalize (object);
}

static void
foundry_input_choice_get_property (GObject    *object,
                                   guint       prop_id,
                                   GValue     *value,
                                   GParamSpec *pspec)
{
  FoundryInputChoice *self = FOUNDRY_INPUT_CHOICE (object);

  switch (prop_id)
    {
    case PROP_ITEM:
      g_value_take_object (value, foundry_input_choice_dup_item (self));
      break;

    case PROP_SELECTED:
      g_value_set_boolean (value, foundry_input_choice_get_selected (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_choice_set_property (GObject      *object,
                                   guint         prop_id,
                                   const GValue *value,
                                   GParamSpec   *pspec)
{
  FoundryInputChoice *self = FOUNDRY_INPUT_CHOICE (object);

  switch (prop_id)
    {
    case PROP_ITEM:
      self->item = g_value_dup_object (value);
      break;

    case PROP_SELECTED:
      foundry_input_choice_set_selected (self, g_value_get_boolean (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_input_choice_class_init (FoundryInputChoiceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_input_choice_finalize;
  object_class->get_property = foundry_input_choice_get_property;
  object_class->set_property = foundry_input_choice_set_property;

  properties[PROP_ITEM] =
    g_param_spec_object ("item", NULL, NULL,
                         G_TYPE_OBJECT,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SELECTED] =
    g_param_spec_boolean ("selected", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_input_choice_init (FoundryInputChoice *self)
{
  g_mutex_init (&self->mutex);
}

FoundryInput *
foundry_input_choice_new (const char *title,
                          const char *subtitle,
                          GObject    *item)
{
  g_return_val_if_fail (!item || G_IS_OBJECT (item), NULL);

  return g_object_new (FOUNDRY_TYPE_INPUT_CHOICE,
                       "title", title,
                       "subtitle", subtitle,
                       "item", item,
                       NULL);
}

gboolean
foundry_input_choice_get_selected (FoundryInputChoice *self)
{
  gboolean ret;

  g_return_val_if_fail (FOUNDRY_IS_INPUT_CHOICE (self), FALSE);

  g_mutex_lock (&self->mutex);
  ret = self->selected;
  g_mutex_unlock (&self->mutex);

  return ret;
}

void
foundry_input_choice_set_selected (FoundryInputChoice *self,
                                   gboolean            selected)
{
  gboolean changed = FALSE;

  g_return_if_fail (FOUNDRY_IS_INPUT_CHOICE (self));

  selected = !!selected;

  g_mutex_lock (&self->mutex);
  if (self->selected != selected)
    {
      self->selected = selected;
      changed = TRUE;
    }
  g_mutex_unlock (&self->mutex);

  if (changed)
    foundry_notify_pspec_in_main (G_OBJECT (self), properties[PROP_SELECTED]);
}

/**
 * foundry_input_choice_dup_item:
 * @self: a [class@Foundry.InputChoice]
 *
 * Gets the item for the choice, if any.
 *
 * Returns: (transfer full) (nullable):
 */
GObject *
foundry_input_choice_dup_item (FoundryInputChoice *self)
{
  GObject *ret;

  g_return_val_if_fail (FOUNDRY_IS_INPUT_CHOICE (self), NULL);

  g_mutex_lock (&self->mutex);
  ret = self->item ? g_object_ref (self->item) : NULL;
  g_mutex_unlock (&self->mutex);

  return ret;
}
