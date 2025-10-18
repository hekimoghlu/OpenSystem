/* plugin-flatpak.h
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

#include <flatpak.h>
#include <foundry.h>
#include <libdex.h>

G_BEGIN_DECLS

DexFuture    *plugin_flatpak_installation_new_system           (void)
  G_GNUC_WARN_UNUSED_RESULT;
DexFuture    *plugin_flatpak_installation_new_user             (void)
  G_GNUC_WARN_UNUSED_RESULT;
DexFuture    *plugin_flatpak_installation_new_private          (FoundryContext      *context)
  G_GNUC_WARN_UNUSED_RESULT;
DexFuture    *plugin_flatpak_installation_new_for_path         (GFile               *path,
                                                                gboolean             user)
  G_GNUC_WARN_UNUSED_RESULT;
DexFuture    *plugin_flatpak_load_installations                (void)
  G_GNUC_WARN_UNUSED_RESULT;
DexFuture    *plugin_flatpak_installation_list_installed_refs  (FoundryContext      *context,
                                                                FlatpakInstallation *installation,
                                                                FlatpakQueryFlags    flags)
  G_GNUC_WARN_UNUSED_RESULT;
DexFuture    *plugin_flatpak_installation_list_refs            (FoundryContext      *context,
                                                                FlatpakInstallation *installation,
                                                                FlatpakQueryFlags    flags)
  G_GNUC_WARN_UNUSED_RESULT;
DexFuture    *plugin_flatpak_installation_list_refs_for_remote (FoundryContext      *context,
                                                                FlatpakInstallation *installation,
                                                                FlatpakRemote       *remote,
                                                                FlatpakQueryFlags    flags)
  G_GNUC_WARN_UNUSED_RESULT;
DexFuture    *plugin_flatpak_find_ref                          (FoundryContext      *context,
                                                                FlatpakInstallation *installation,
                                                                const char          *runtime,
                                                                const char          *arch,
                                                                const char          *runtime_version)
  G_GNUC_WARN_UNUSED_RESULT;
FlatpakRemote *plugin_flatpak_find_remote                      (FoundryContext      *context,
                                                                FlatpakInstallation *installation,
                                                                FlatpakRef          *ref)
  G_GNUC_WARN_UNUSED_RESULT;
gboolean       plugin_flatpak_ref_can_be_sdk                   (FlatpakRef          *ref);
gboolean       plugin_flatpak_ref_matches                      (FlatpakRef          *ref,
                                                                const char          *name,
                                                                const char          *arch,
                                                                const char          *branch);
void           plugin_flatpak_apply_config_dir                 (FoundryContext         *context,
                                                                FoundryProcessLauncher *launcher);

G_END_DECLS
