/* foundry-flatpak-arch-options.c
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

#include "foundry-flatpak-arch-options.h"
#include "foundry-flatpak-serializable-private.h"

struct _FoundryFlatpakArchOptions
{
  FoundryFlatpakSerializable parent_instance;
  GHashTable *arches;
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakArchOptions, foundry_flatpak_arch_options, FOUNDRY_TYPE_FLATPAK_SERIALIZABLE)

static DexFuture *
foundry_flatpak_arch_options_deserialize (FoundryFlatpakSerializable *serializable,
                                          JsonNode                   *node)
{
  FoundryFlatpakArchOptions *self = (FoundryFlatpakArchOptions *)serializable;
  g_autoptr(GFile) base_dir = NULL;
  JsonObject *object;
  JsonObjectIter iter;
  const char *member_name;
  JsonNode *member_node;

  g_assert (FOUNDRY_IS_FLATPAK_ARCH_OPTIONS (self));
  g_assert (node != NULL);

  if (!JSON_NODE_HOLDS_OBJECT (node))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_FAILED,
                                  "Expected object for arch");

  base_dir = _foundry_flatpak_serializable_dup_base_dir (serializable);
  object = json_node_get_object (node);

  json_object_iter_init (&iter, object);
  while (json_object_iter_next (&iter, &member_name, &member_node))
    {
      g_autoptr(FoundryFlatpakSerializable) options = NULL;
      g_autoptr(GError) error = NULL;

      options = _foundry_flatpak_serializable_new (FOUNDRY_TYPE_FLATPAK_OPTIONS, base_dir);

      if (!dex_await (_foundry_flatpak_serializable_deserialize (options, member_node), &error))
        return dex_future_new_for_error (g_steal_pointer (&error));

      g_hash_table_replace (self->arches, g_strdup (member_name), g_steal_pointer (&options));
    }

  return dex_future_new_true ();
}

static void
foundry_flatpak_arch_options_finalize (GObject *object)
{
  FoundryFlatpakArchOptions *self = (FoundryFlatpakArchOptions *)object;

  g_clear_pointer (&self->arches, g_hash_table_unref);

  G_OBJECT_CLASS (foundry_flatpak_arch_options_parent_class)->finalize (object);
}

static void
foundry_flatpak_arch_options_class_init (FoundryFlatpakArchOptionsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryFlatpakSerializableClass *serializable_class = FOUNDRY_FLATPAK_SERIALIZABLE_CLASS (klass);

  object_class->finalize = foundry_flatpak_arch_options_finalize;

  serializable_class->deserialize = foundry_flatpak_arch_options_deserialize;
}

static void
foundry_flatpak_arch_options_init (FoundryFlatpakArchOptions *self)
{
  self->arches = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, g_object_unref);
}

/**
 * foundry_flatpak_arch_options_dup_arches:
 * @self: a [class@Foundry.FlatpakArchOptions]
 *
 * Returns: (transfer full) (nullable):
 */
char **
foundry_flatpak_arch_options_dup_arches (FoundryFlatpakArchOptions *self)
{
  g_auto(GStrv) keys = NULL;

  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_ARCH_OPTIONS (self), NULL);

  keys = (char **)g_hash_table_get_keys_as_array (self->arches, NULL);
  for (gsize i = 0; keys[i]; i++)
    keys[i] = g_strdup (keys[i]);

  return g_steal_pointer (&keys);
}

/**
 * foundry_flatpak_arch_options_dup_arch:
 * @self: a [class@Foundry.FlatpakArchOptions]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryFlatpakOptions *
foundry_flatpak_arch_options_dup_arch (FoundryFlatpakArchOptions *self,
                                       const char                *arch)
{
  FoundryFlatpakOptions *ret;

  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_ARCH_OPTIONS (self), NULL);
  g_return_val_if_fail (arch != NULL, NULL);

  if ((ret = g_hash_table_lookup (self->arches, arch)))
    return g_object_ref (ret);

  return NULL;
}
