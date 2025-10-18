/* foundry-terminal-palette-set.c
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

#include "foundry-terminal-palette-private.h"
#include "foundry-terminal-palette-set.h"

struct _FoundryTerminalPaletteSet
{
  GObject     parent_instance;
  GHashTable *palettes;
  char       *title;
};

enum {
  PROP_0,
  PROP_DARK,
  PROP_LIGHT,
  PROP_TITLE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryTerminalPaletteSet, foundry_terminal_palette_set, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_terminal_palette_set_finalize (GObject *object)
{
  FoundryTerminalPaletteSet *self = (FoundryTerminalPaletteSet *)object;

  g_clear_pointer (&self->palettes, g_hash_table_unref);
  g_clear_pointer (&self->title, g_free);

  G_OBJECT_CLASS (foundry_terminal_palette_set_parent_class)->finalize (object);
}

static void
foundry_terminal_palette_set_get_property (GObject    *object,
                                           guint       prop_id,
                                           GValue     *value,
                                           GParamSpec *pspec)
{
  FoundryTerminalPaletteSet *self = FOUNDRY_TERMINAL_PALETTE_SET (object);

  switch (prop_id)
    {
    case PROP_DARK:
      g_value_take_object (value, foundry_terminal_palette_set_dup_dark (self));
      break;

    case PROP_LIGHT:
      g_value_take_object (value, foundry_terminal_palette_set_dup_light (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, foundry_terminal_palette_set_dup_title (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_terminal_palette_set_class_init (FoundryTerminalPaletteSetClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_terminal_palette_set_finalize;
  object_class->get_property = foundry_terminal_palette_set_get_property;

  properties[PROP_LIGHT] =
    g_param_spec_object ("light", NULL, NULL,
                         FOUNDRY_TYPE_TERMINAL_PALETTE,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_DARK] =
    g_param_spec_object ("dark", NULL, NULL,
                         FOUNDRY_TYPE_TERMINAL_PALETTE,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_terminal_palette_set_init (FoundryTerminalPaletteSet *self)
{
}

static DexFuture *
foundry_terminal_palette_set_new_fiber (gpointer data)
{
  g_autoptr(FoundryTerminalPaletteSet) self = NULL;
  g_autoptr(GHashTable) hash = NULL;
  g_autoptr(GKeyFile) key_file = NULL;
  g_autoptr(GError) error = NULL;
  GBytes *bytes = data;
  g_auto(GStrv) groups = NULL;
  g_autofree char *title = NULL;

  g_assert (bytes != NULL);

  key_file = g_key_file_new ();

  if (!g_key_file_load_from_bytes (key_file, bytes, 0, &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  hash = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, g_object_unref);
  groups = g_key_file_get_groups (key_file, NULL);
  title = g_key_file_get_string (key_file, "Palette", "Name", NULL);

  for (gsize i = 0; groups[i]; i++)
    {
      g_autoptr(FoundryTerminalPalette) palette = NULL;

      if (g_str_equal (groups[i], "Palette"))
        continue;

      if (!(palette = _foundry_terminal_palette_new (title, key_file, groups[i], &error)))
        return dex_future_new_for_error (g_steal_pointer (&error));

      g_hash_table_replace (hash,
                            g_utf8_strdown (groups[i], -1),
                            g_steal_pointer (&palette));
    }

  if (g_hash_table_size (hash) == 0)
    {
      g_autoptr(FoundryTerminalPalette) palette = NULL;

      /* Try to parse the palette from the main "Palette" group */
      if (!(palette = _foundry_terminal_palette_new (title, key_file, "Palette", &error)))
        return dex_future_new_reject (G_IO_ERROR,
                                      G_IO_ERROR_INVALID_DATA,
                                      "No palettes defined");

      g_hash_table_replace (hash,
                            g_strdup ("any"),
                            g_steal_pointer (&palette));
    }

  self = g_object_new (FOUNDRY_TYPE_TERMINAL_PALETTE_SET, NULL);
  self->palettes = g_steal_pointer (&hash);
  self->title = g_steal_pointer (&title);

  return dex_future_new_take_object (g_steal_pointer (&self));
}

/**
 * foundry_terminal_palette_set_new:
 * @bytes:
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [class@FoundryGtk.TerminalPaletteSet] or rejects with error.
 */
DexFuture *
foundry_terminal_palette_set_new (GBytes *bytes)
{
  g_return_val_if_fail (bytes != NULL, NULL);

  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              foundry_terminal_palette_set_new_fiber,
                              g_bytes_ref (bytes),
                              (GDestroyNotify) g_bytes_unref);
}

static FoundryTerminalPalette *
lookup_with_fallback (FoundryTerminalPaletteSet *self,
                      const char                *first,
                      const char                *fallback)
{
  FoundryTerminalPalette *palette;
  GHashTableIter iter;
  gpointer key, value;

  g_assert (FOUNDRY_IS_TERMINAL_PALETTE_SET (self));
  g_assert (first != NULL);
  g_assert (fallback != NULL);

  if ((palette = g_hash_table_lookup (self->palettes, first)))
    return g_object_ref (palette);

  if ((palette = g_hash_table_lookup (self->palettes, fallback)))
    return g_object_ref (palette);

  g_hash_table_iter_init (&iter, self->palettes);

  if (g_hash_table_iter_next (&iter, &key, &value))
    return g_object_ref (value);

  g_assert_not_reached ();

  return NULL;
}

/**
 * foundry_terminal_palette_set_dup_light:
 * @self: a [class@FoundryGtk.TerminalPaletteSet]
 *
 * Returns: (transfer full):
 */
FoundryTerminalPalette *
foundry_terminal_palette_set_dup_light (FoundryTerminalPaletteSet *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TERMINAL_PALETTE_SET (self), NULL);

  return lookup_with_fallback (self, "light", "dark");
}

/**
 * foundry_terminal_palette_set_dup_dark:
 * @self: a [class@FoundryGtk.TerminalPaletteSet]
 *
 * Returns: (transfer full):
 */
FoundryTerminalPalette *
foundry_terminal_palette_set_dup_dark (FoundryTerminalPaletteSet *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TERMINAL_PALETTE_SET (self), NULL);

  return lookup_with_fallback (self, "dark", "light");
}

char *
foundry_terminal_palette_set_dup_title (FoundryTerminalPaletteSet *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TERMINAL_PALETTE_SET (self), NULL);

  return g_strdup (self->title);
}
