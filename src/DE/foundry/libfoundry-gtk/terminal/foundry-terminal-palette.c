/* foundry-terminal-palette.c
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

struct _FoundryTerminalPalette
{
  GObject  parent_instance;
  char    *title;
  GdkRGBA  colors[16];
  GdkRGBA  background;
  GdkRGBA  foreground;
  GdkRGBA  cursor_background;
  GdkRGBA  cursor_foreground;
  guint    background_set : 1;
  guint    foreground_set : 1;
  guint    cursor_background_set : 1;
  guint    cursor_foreground_set : 1;
};

G_DEFINE_FINAL_TYPE (FoundryTerminalPalette, foundry_terminal_palette, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_TITLE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_terminal_palette_finalize (GObject *object)
{
  FoundryTerminalPalette *self = (FoundryTerminalPalette *)object;

  g_clear_pointer (&self->title, g_free);

  G_OBJECT_CLASS (foundry_terminal_palette_parent_class)->finalize (object);
}

static void
foundry_terminal_palette_get_property (GObject    *object,
                                       guint       prop_id,
                                       GValue     *value,
                                       GParamSpec *pspec)
{
  FoundryTerminalPalette *self = FOUNDRY_TERMINAL_PALETTE (object);

  switch (prop_id)
    {
    case PROP_TITLE:
      g_value_set_string (value, self->title);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_terminal_palette_class_init (FoundryTerminalPaletteClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_terminal_palette_finalize;
  object_class->get_property = foundry_terminal_palette_get_property;

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_terminal_palette_init (FoundryTerminalPalette *self)
{
}

static gboolean
get_color (GKeyFile   *key_file,
           const char *group,
           const char *name,
           GdkRGBA    *rgba)
{
  g_autofree char *str = NULL;

  if (!g_key_file_has_key (key_file, group, name, NULL))
    return FALSE;

  if (!(str = g_key_file_get_string (key_file, group, name, NULL)))
    return FALSE;

  return gdk_rgba_parse (rgba, str);
}

FoundryTerminalPalette *
_foundry_terminal_palette_new (const char  *title,
                               GKeyFile    *key_file,
                               const char  *group,
                               GError     **error)
{
  g_autoptr(FoundryTerminalPalette) self = NULL;

  g_assert (key_file != NULL);
  g_assert (group != NULL);

  self = g_object_new (FOUNDRY_TYPE_TERMINAL_PALETTE, NULL);

  self->title = g_strdup (title);
  self->foreground_set = get_color (key_file, group, "Foreground", &self->foreground);
  self->background_set = get_color (key_file, group, "Background", &self->background);
  self->cursor_foreground_set = get_color (key_file, group, "CursorForeground", &self->cursor_foreground);
  self->cursor_background_set = get_color (key_file, group, "CursorBackground", &self->cursor_background);

  for (guint i = 0; i < G_N_ELEMENTS (self->colors); i++)
    {
      char key[32];

      g_snprintf (key, sizeof key, "Color%u", i);

      if (!get_color (key_file, group, key, &self->colors[i]))
        return FALSE;
    }

  return g_steal_pointer (&self);
}

void
_foundry_terminal_palette_apply (FoundryTerminalPalette *self,
                                 VteTerminal            *terminal)
{
  GdkRGBA *foreground = NULL;
  GdkRGBA *background = NULL;

  g_return_if_fail (FOUNDRY_IS_TERMINAL_PALETTE (self));
  g_return_if_fail (VTE_IS_TERMINAL (terminal));

  if (self->foreground_set)
    foreground = &self->foreground;

  if (self->background_set)
    background = &self->background;

  vte_terminal_set_colors (terminal,
                           foreground,
                           background,
                           self->colors,
                           G_N_ELEMENTS (self->colors));

  if (self->cursor_background_set)
    vte_terminal_set_color_cursor (terminal, &self->cursor_background);

  if (self->cursor_foreground_set)
    vte_terminal_set_color_cursor_foreground (terminal, &self->cursor_foreground);
}
