/* libspelling.c
 *
 * Copyright 2023 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include <glib/gi18n.h>

#include "spelling-checker.h"
#include "spelling-dictionary.h"
#include "spelling-init.h"
#include "spelling-language.h"
#include "spelling-provider.h"
#include "spelling-text-buffer-adapter.h"

#include "gconstructor.h"

G_DEFINE_CONSTRUCTOR (_spelling_init)

static void
_spelling_init (void)
{
  static gsize initialized;

  if (g_once_init_enter (&initialized))
    {
      bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");
      bindtextdomain (GETTEXT_PACKAGE, PACKAGE_LOCALE_DIR);

      g_type_ensure (SPELLING_TYPE_CHECKER);
      g_type_ensure (SPELLING_TYPE_DICTIONARY);
      g_type_ensure (SPELLING_TYPE_LANGUAGE);
      g_type_ensure (SPELLING_TYPE_PROVIDER);
      g_type_ensure (SPELLING_TYPE_TEXT_BUFFER_ADAPTER);

      g_once_init_leave (&initialized, TRUE);
    }
}

/**
 * spelling_init:
 *
 * Call this function before using any other libspelling functions in your
 * applications. It will initialize everything needed to operate the library.
 */
void
spelling_init (void)
{
  _spelling_init ();
}
