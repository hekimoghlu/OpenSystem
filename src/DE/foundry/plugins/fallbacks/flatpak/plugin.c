/* plugin.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include "plugin-flatpak-build-addin.h"
#include "plugin-flatpak-config-provider.h"
#include "plugin-flatpak-dependency-provider.h"
#include "plugin-flatpak-sdk-provider.h"

#ifdef FOUNDRY_FEATURE_DOCS
# include "plugin-flatpak-documentation-provider.h"
#endif

FOUNDRY_PLUGIN_DEFINE (_plugin_flatpak_register_types,
                       FOUNDRY_PLUGIN_REGISTER_TYPE (FOUNDRY_TYPE_BUILD_ADDIN, PLUGIN_TYPE_FLATPAK_BUILD_ADDIN)
                       FOUNDRY_PLUGIN_REGISTER_TYPE (FOUNDRY_TYPE_CONFIG_PROVIDER, PLUGIN_TYPE_FLATPAK_CONFIG_PROVIDER)
                       FOUNDRY_PLUGIN_REGISTER_TYPE (FOUNDRY_TYPE_DEPENDENCY_PROVIDER, PLUGIN_TYPE_FLATPAK_DEPENDENCY_PROVIDER)
#ifdef FOUNDRY_FEATURE_DOCS
                       FOUNDRY_PLUGIN_REGISTER_TYPE (FOUNDRY_TYPE_DOCUMENTATION_PROVIDER, PLUGIN_TYPE_FLATPAK_DOCUMENTATION_PROVIDER)
#endif
                       FOUNDRY_PLUGIN_REGISTER_TYPE (FOUNDRY_TYPE_SDK_PROVIDER, PLUGIN_TYPE_FLATPAK_SDK_PROVIDER))
