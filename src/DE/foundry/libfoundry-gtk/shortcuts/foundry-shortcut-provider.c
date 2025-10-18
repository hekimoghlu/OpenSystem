/* foundry-shortcut-provider.c
 *
 * Copyright 2022-2025 Christian Hergert <chergert@redhat.com>
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

#include "foundry-shortcut-provider.h"

G_DEFINE_INTERFACE (FoundryShortcutProvider, foundry_shortcut_provider, FOUNDRY_TYPE_CONTEXTUAL)

static GListModel *
foundry_shortcut_provider_real_list_shortcuts (FoundryShortcutProvider *self)
{
  return G_LIST_MODEL (g_list_store_new (GTK_TYPE_SHORTCUT));
}

static void
foundry_shortcut_provider_default_init (FoundryShortcutProviderInterface *iface)
{
  iface->list_shortcuts = foundry_shortcut_provider_real_list_shortcuts;
}

/**
 * foundry_shortcut_provider_list_shortcuts:
 * @self: a #FoundryShortcutProvider
 *
 * Gets a #GListModel of #GtkShortcut.
 *
 * This function should return a #GListModel of #GtkShortcut that are updated
 * as necessary by the plugin. This list model is used to activate shortcuts
 * based on user input and allows more control by plugins over when and how
 * shortcuts may activate.
 *
 * Returns: (transfer full): A #GListModel of #GtkShortcut
 */
GListModel *
foundry_shortcut_provider_list_shortcuts (FoundryShortcutProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_SHORTCUT_PROVIDER (self), NULL);

  return FOUNDRY_SHORTCUT_PROVIDER_GET_IFACE (self)->list_shortcuts (self);
}

