/* foundry-adw-init.c
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

#include <adwaita.h>

#include "foundry-adw-init.h"
#include "foundry-adw-resources.h"

#include "foundry-page.h"
#include "foundry-panel.h"
#include "foundry-search-dialog.h"
#include "foundry-workspace.h"
#include "foundry-workspace-addin.h"

static void
_foundry_adw_init_once (void)
{
  g_autoptr(GtkCssProvider) css_provider = NULL;
  GdkDisplay *display;

  g_resources_register (_foundry_adw_get_resource ());

  if (!(display = gdk_display_get_default ()))
    {
      g_debug ("No GDK display, skipping full initialization");
      return;
    }

  foundry_gtk_init ();
  adw_init ();

  g_type_ensure (FOUNDRY_TYPE_PAGE);
  g_type_ensure (FOUNDRY_TYPE_PANEL);
  g_type_ensure (FOUNDRY_TYPE_SEARCH_DIALOG);
  g_type_ensure (FOUNDRY_TYPE_WORKSPACE);
  g_type_ensure (FOUNDRY_TYPE_WORKSPACE_ADDIN);

  css_provider = gtk_css_provider_new ();
  gtk_css_provider_load_from_resource (css_provider, "/app/devsuite/foundry-adw/style.css");
  gtk_style_context_add_provider_for_display (display,
                                              GTK_STYLE_PROVIDER (css_provider),
                                              GTK_STYLE_PROVIDER_PRIORITY_USER + 1);
}

void
foundry_adw_init (void)
{
  static gsize initialized;

  if (g_once_init_enter (&initialized))
    {
      _foundry_adw_init_once ();
      g_once_init_leave (&initialized, TRUE);
    }
}
