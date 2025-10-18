/* plugin.c
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

#include <foundry.h>

#include "plugin-ctags-completion-provider.h"
#include "plugin-ctags-service.h"
#include "plugin-ctags-symbol-provider.h"

FOUNDRY_PLUGIN_DEFINE (_plugin_ctags_register_types,
                       FOUNDRY_PLUGIN_REGISTER_TYPE (FOUNDRY_TYPE_COMPLETION_PROVIDER, PLUGIN_TYPE_CTAGS_COMPLETION_PROVIDER)
                       FOUNDRY_PLUGIN_REGISTER_TYPE (FOUNDRY_TYPE_SERVICE, PLUGIN_TYPE_CTAGS_SERVICE)
                       FOUNDRY_PLUGIN_REGISTER_TYPE (FOUNDRY_TYPE_SYMBOL_PROVIDER, PLUGIN_TYPE_CTAGS_SYMBOL_PROVIDER))
