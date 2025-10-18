/* foundry-terminal.c
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

#include "foundry-terminal.h"
#include "foundry-terminal-palette-private.h"
#include "foundry-terminal-palette-set.h"

typedef struct
{
  FoundryTerminalPalette *palette;
} FoundryTerminalPrivate;

enum {
  PROP_0,
  PROP_PALETTE,
  N_PROPS
};

G_DEFINE_TYPE_WITH_PRIVATE (FoundryTerminal, foundry_terminal, VTE_TYPE_TERMINAL)

static GParamSpec *properties[N_PROPS];
static GSettings *terminal_settings;

static void
foundry_terminal_update_scrollback (FoundryTerminal *self)
{
  guint max_scrollback_lines = 10000000;

  g_assert (FOUNDRY_IS_TERMINAL (self));

  if (g_settings_get_boolean (terminal_settings, "limit-scrollback"))
    max_scrollback_lines = g_settings_get_uint (terminal_settings, "max-scrollback-lines");

  vte_terminal_set_scrollback_lines (VTE_TERMINAL (self), max_scrollback_lines);
}

static void
foundry_terminal_update_font (FoundryTerminal *self)
{
  g_autofree char *custom_font = NULL;
  g_autoptr(PangoFontDescription) font_desc = NULL;

  g_assert (FOUNDRY_IS_TERMINAL (self));

  if (g_settings_get_boolean (terminal_settings, "use-custom-font"))
    custom_font = g_settings_get_string (terminal_settings, "custom-font");

  if (custom_font != NULL)
    font_desc = pango_font_description_from_string (custom_font);

  vte_terminal_set_font (VTE_TERMINAL (self), font_desc);
}

static void
foundry_terminal_finalize (GObject *object)
{
  FoundryTerminal *self = (FoundryTerminal *)object;
  FoundryTerminalPrivate *priv = foundry_terminal_get_instance_private (self);

  g_clear_object (&priv->palette);

  G_OBJECT_CLASS (foundry_terminal_parent_class)->finalize (object);
}

static void
foundry_terminal_get_property (GObject    *object,
                               guint       prop_id,
                               GValue     *value,
                               GParamSpec *pspec)
{
  FoundryTerminal *self = FOUNDRY_TERMINAL (object);

  switch (prop_id)
    {
    case PROP_PALETTE:
      g_value_set_object (value, foundry_terminal_get_palette (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_terminal_set_property (GObject      *object,
                               guint         prop_id,
                               const GValue *value,
                               GParamSpec   *pspec)
{
  FoundryTerminal *self = FOUNDRY_TERMINAL (object);

  switch (prop_id)
    {
    case PROP_PALETTE:
      foundry_terminal_set_palette (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_terminal_class_init (FoundryTerminalClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_terminal_finalize;
  object_class->get_property = foundry_terminal_get_property;
  object_class->set_property = foundry_terminal_set_property;

  properties[PROP_PALETTE] =
    g_param_spec_object ("palette", NULL, NULL,
                         FOUNDRY_TYPE_TERMINAL_PALETTE,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_terminal_init (FoundryTerminal *self)
{
  if (terminal_settings == NULL)
    terminal_settings = g_settings_new ("app.devsuite.foundry.terminal");

  g_settings_bind (terminal_settings, "allow-bold",
                   self, "allow-bold",
                   G_SETTINGS_BIND_DEFAULT);
  g_settings_bind (terminal_settings, "allow-hyperlink",
                   self, "allow-hyperlink",
                   G_SETTINGS_BIND_DEFAULT);
  g_settings_bind (terminal_settings, "scroll-on-output",
                   self, "scroll-on-output",
                   G_SETTINGS_BIND_DEFAULT);
  g_settings_bind (terminal_settings, "scroll-on-keystroke",
                   self, "scroll-on-keystroke",
                   G_SETTINGS_BIND_DEFAULT);

  g_signal_connect_object (terminal_settings,
                           "changed::limit-scrollback",
                           G_CALLBACK (foundry_terminal_update_scrollback),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (terminal_settings,
                           "changed::max-scrollback-lines",
                           G_CALLBACK (foundry_terminal_update_scrollback),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (terminal_settings,
                           "changed::use-custom-font",
                           G_CALLBACK (foundry_terminal_update_font),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (terminal_settings,
                           "changed::custom-font",
                           G_CALLBACK (foundry_terminal_update_font),
                           self,
                           G_CONNECT_SWAPPED);

  foundry_terminal_update_font (self);
  foundry_terminal_update_scrollback (self);
}

GtkWidget *
foundry_terminal_new (void)
{
  return g_object_new (FOUNDRY_TYPE_TERMINAL, NULL);
}

/**
 * foundry_terminal_get_palette:
 * @self: a [class@FoundryGtk.Terminal]
 *
 * Returns: (transfer none) (nullable):
 */
FoundryTerminalPalette *
foundry_terminal_get_palette (FoundryTerminal *self)
{
  FoundryTerminalPrivate *priv = foundry_terminal_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_TERMINAL (self), NULL);

  return priv->palette;
}

void
foundry_terminal_set_palette (FoundryTerminal        *self,
                              FoundryTerminalPalette *palette)
{
  FoundryTerminalPrivate *priv = foundry_terminal_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_TERMINAL (self));
  g_return_if_fail (!palette || FOUNDRY_IS_TERMINAL_PALETTE (palette));

  if (g_set_object (&priv->palette, palette))
    {
      if (priv->palette != NULL)
        _foundry_terminal_palette_apply (priv->palette, VTE_TERMINAL (self));
      else
        vte_terminal_set_default_colors (VTE_TERMINAL (self));

      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_PALETTE]);
    }
}

static DexFuture *
foundry_terminal_list_palettes_fiber (gpointer user_data)
{
  const char *base_path = "/app/devsuite/foundry/terminal/palettes";
  g_auto(GStrv) children = g_resources_enumerate_children (base_path, 0, NULL);
  g_autoptr(GListStore) store = g_list_store_new (FOUNDRY_TYPE_TERMINAL_PALETTE_SET);

  for (gsize i = 0; children[i]; i++)
    {
      g_autofree char *path = g_build_filename (base_path, children[i], NULL);
      g_autoptr(GBytes) bytes = g_resources_lookup_data (path, 0, NULL);

      if (bytes != NULL)
        {
          g_autoptr(FoundryTerminalPaletteSet) set = NULL;
          g_autoptr(GError) error = NULL;

          if (!(set = dex_await_object (foundry_terminal_palette_set_new (bytes), &error)))
            g_warning ("Failed to parse `%s`: %s", path, error->message);
          else
            g_list_store_append (store, set);
        }
    }

  return dex_future_new_take_object (g_steal_pointer (&store));
}

/**
 * foundry_terminal_list_palette_sets:
 *
 * Lists available palettes known to Foundry.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@FoundryGtk.TerminalPaletteSet].
 */
DexFuture *
foundry_terminal_list_palette_sets (void)
{
  static DexFuture *future;

  if (future == NULL)
    {
      future = dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                                    foundry_terminal_list_palettes_fiber,
                                    NULL, NULL);
      dex_future_disown (dex_ref (future));
    }

  return dex_ref (future);
}

static DexFuture *
foundry_terminal_find_palette_cb (DexFuture *completed,
                                  gpointer   user_data)
{
  g_autoptr(GListModel) model = NULL;
  const char *name = user_data;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (name != NULL);

  if ((model = dex_await_object (dex_ref (completed), NULL)))
    {
      guint n_items = g_list_model_get_n_items (model);

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryTerminalPaletteSet) set = g_list_model_get_item (model, i);
          g_autofree char *title = foundry_terminal_palette_set_dup_title (set);

          if (g_strcmp0 (name, title) == 0)
            return dex_future_new_take_object (g_steal_pointer (&set));
        }
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Failed to locate palette named `%s`",
                                name);
}

/**
 * foundry_terminal_find_palette_set:
 * @name: the palette name
 *
 * Tries to locate a palette by name.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@FoundryGtk.TerminalPaletteSet] or rejects with error.
 */
DexFuture *
foundry_terminal_find_palette_set (const char *name)
{
  dex_return_error_if_fail (name != NULL);

  return dex_future_then (foundry_terminal_list_palette_sets (),
                          foundry_terminal_find_palette_cb,
                          g_strdup (name),
                          g_free);
}
