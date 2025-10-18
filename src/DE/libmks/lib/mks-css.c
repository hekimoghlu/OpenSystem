/*
 * mks-css.c
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include <gtk/gtk.h>

#include "mks-css-private.h"

void
_mks_css_init (void)
{
  static GtkCssProvider *provider;

  if G_UNLIKELY (provider == NULL)
    {
      GdkDisplay *display = gdk_display_get_default ();

      if (display == NULL)
        return;

      provider = gtk_css_provider_new ();
      gtk_css_provider_load_from_resource (provider, "/org/gnome/libmks/style.css");
      gtk_style_context_add_provider_for_display (display,
                                                  GTK_STYLE_PROVIDER (provider),
                                                  GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    }
}
