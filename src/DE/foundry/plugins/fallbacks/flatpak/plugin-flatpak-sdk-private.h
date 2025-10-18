/* plugin-flatpak-sdk-private.h
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

#pragma once

#include "plugin-flatpak-sdk.h"

G_BEGIN_DECLS

struct _PluginFlatpakSdk
{
  FoundrySdk           parent_instance;
  FlatpakInstallation *installation;
  FlatpakRef          *ref;
  FoundryPathCache    *path_cache;
};

DexFuture *plugin_flatpak_sdk_install (FoundrySdk          *sdk,
                                       FoundryOperation    *operation,
                                       DexCancellable      *cancellable) G_GNUC_WARN_UNUSED_RESULT;
DexFuture *plugin_flatpak_ref_install (FoundryContext      *context,
                                       FlatpakInstallation *installation,
                                       FlatpakRef          *ref,
                                       FoundryOperation    *operation,
                                       gboolean             is_installed,
                                       DexCancellable      *cancellable) G_GNUC_WARN_UNUSED_RESULT;

G_END_DECLS
