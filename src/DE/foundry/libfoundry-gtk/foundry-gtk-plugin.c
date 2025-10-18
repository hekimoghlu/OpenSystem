/* foundry-gtk-plugin.c
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

#include <libpeas.h>

#include <foundry.h>

#include "foundry-gtk-resources.h"
#include "foundry-gtk-tweak-provider-private.h"
#include "foundry-source-buffer.h"
#include "foundry-source-buffer-provider-private.h"
#include "foundry-source-language-guesser.h"
#include "foundry-version-macros.h"

FOUNDRY_AVAILABLE_IN_ALL
void _foundry_gtk_register_types (PeasObjectModule *module);

void
_foundry_gtk_register_types (PeasObjectModule *module)
{
  g_resources_register (_foundry_gtk_get_resource ());

  peas_object_module_register_extension_type (module,
                                              FOUNDRY_TYPE_TEXT_BUFFER_PROVIDER,
                                              FOUNDRY_TYPE_SOURCE_BUFFER_PROVIDER);
  peas_object_module_register_extension_type (module,
                                              FOUNDRY_TYPE_LANGUAGE_GUESSER,
                                              FOUNDRY_TYPE_SOURCE_LANGUAGE_GUESSER);
  peas_object_module_register_extension_type (module,
                                              FOUNDRY_TYPE_TWEAK_PROVIDER,
                                              FOUNDRY_TYPE_GTK_TWEAK_PROVIDER);
}
