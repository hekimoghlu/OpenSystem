/* foundry-input-group.c
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

#include "foundry-input-group.h"
#include "foundry-input-validator-delegate.h"
#include "foundry-util-private.h"

struct _FoundryInputGroup
{
  FoundryInput  parent_instance;
  GListStore   *children;
};

G_DEFINE_FINAL_TYPE (FoundryInputGroup, foundry_input_group, FOUNDRY_TYPE_INPUT)

static DexFuture *
foundry_input_group_validate (FoundryInput *input,
                              gpointer      user_data)
{
  GWeakRef *wr = user_data;
  g_autoptr(FoundryInputGroup) self = g_weak_ref_get (wr);
  guint n_items;

  g_assert (!self || FOUNDRY_IS_INPUT_GROUP (self));

  if (self != NULL &&
      self->children != NULL &&
      (n_items = g_list_model_get_n_items (G_LIST_MODEL (self->children))) > 0)
    {
      g_autoptr(GPtrArray) futures = g_ptr_array_new_with_free_func (dex_unref);

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryInput) item = g_list_model_get_item (G_LIST_MODEL (self->children), i);

          g_ptr_array_add (futures, foundry_input_validate (item));
        }

      return foundry_future_all (futures);
    }

  return dex_future_new_true ();
}

static void
foundry_input_group_dispose (GObject *object)
{
  FoundryInputGroup *self = (FoundryInputGroup *)object;

  g_clear_object (&self->children);

  G_OBJECT_CLASS (foundry_input_group_parent_class)->dispose (object);
}

static void
foundry_input_group_class_init (FoundryInputGroupClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_input_group_dispose;
}

static void
foundry_input_group_init (FoundryInputGroup *self)
{
  self->children = g_list_store_new (FOUNDRY_TYPE_INPUT);
}

/**
 * foundry_input_group_new:
 * @title: (nullable):
 * @subtitle: (nullable):
 * @validator: (transfer full) (nullable): optional validator
 * @children: (array length=n_children):
 *
 * If @validator is %NULL, then all children will be validated
 * when validation of the group is requested.
 *
 * Returns: (transfer full):
 */
FoundryInput *
foundry_input_group_new (const char             *title,
                         const char             *subtitle,
                         FoundryInputValidator  *validator,
                         FoundryInput          **children,
                         guint                   n_children)
{
  g_autoptr(FoundryInputValidator) stolen = NULL;
  FoundryInputGroup *self;
  GWeakRef *wr = NULL;

  g_return_val_if_fail (children != NULL, NULL);
  g_return_val_if_fail (n_children > 0, NULL);
  g_return_val_if_fail (!validator || FOUNDRY_IS_INPUT_VALIDATOR (validator), NULL);

  if (validator == NULL)
    {
      wr = foundry_weak_ref_new (NULL);
      validator = foundry_input_validator_delegate_new (foundry_input_group_validate,
                                                        wr,
                                                        (GDestroyNotify) foundry_weak_ref_free);
    }

  stolen = validator;

  self = g_object_new (FOUNDRY_TYPE_INPUT_GROUP,
                       "title", title,
                       "subtitle", subtitle,
                       "validator", validator,
                       NULL);

  g_list_store_splice (self->children, 0, 0, (gpointer *)children, n_children);

  if (wr != NULL)
    g_weak_ref_set (wr, self);

  return FOUNDRY_INPUT (self);
}

/**
 * foundry_input_group_list_children:
 * @self: a [class@Foundry.InputGroup]
 *
 * Returns: (transfer full):
 */
GListModel *
foundry_input_group_list_children (FoundryInputGroup *self)
{
  g_return_val_if_fail (FOUNDRY_IS_INPUT_GROUP (self), NULL);

  return g_object_ref (G_LIST_MODEL (self->children));
}
