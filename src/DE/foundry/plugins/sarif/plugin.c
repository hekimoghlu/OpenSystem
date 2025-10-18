/* plugin.c
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include <foundry.h>

#include "plugin-sarif-diagnostic-provider.h"
#include "plugin-sarif-build-addin.h"
#include "plugin-sarif-service.h"

FOUNDRY_PLUGIN_DEFINE (_plugin_sarif_register_types,
                       FOUNDRY_PLUGIN_REGISTER_TYPE (FOUNDRY_TYPE_DIAGNOSTIC_PROVIDER, PLUGIN_TYPE_SARIF_DIAGNOSTIC_PROVIDER)
                       FOUNDRY_PLUGIN_REGISTER_TYPE (FOUNDRY_TYPE_BUILD_ADDIN, PLUGIN_TYPE_SARIF_BUILD_ADDIN)
                       FOUNDRY_PLUGIN_REGISTER_TYPE (FOUNDRY_TYPE_SERVICE, PLUGIN_TYPE_SARIF_SERVICE))
