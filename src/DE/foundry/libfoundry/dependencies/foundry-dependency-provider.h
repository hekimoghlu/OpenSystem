/* foundry-dependency-provider.h
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

#pragma once

#include <libpeas.h>

#include "foundry-config.h"
#include "foundry-contextual.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEPENDENCY_PROVIDER (foundry_dependency_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDependencyProvider, foundry_dependency_provider, FOUNDRY, DEPENDENCY_PROVIDER, FoundryContextual)

struct _FoundryDependencyProviderClass
{
  FoundryContextualClass parent_class;

  DexFuture *(*load)                (FoundryDependencyProvider *self);
  DexFuture *(*unload)              (FoundryDependencyProvider *self);
  DexFuture *(*list_dependencies)   (FoundryDependencyProvider *self,
                                     FoundryConfig             *config,
                                     FoundryDependency         *parent);
  DexFuture *(*update_dependencies) (FoundryDependencyProvider *self,
                                     FoundryConfig             *config,
                                     GListModel                *dependencies,
                                     int                        pty_fd,
                                     DexCancellable            *cancellable);

  /*< private >*/
  gpointer _reserved[12];
};

FOUNDRY_AVAILABLE_IN_ALL
PeasPluginInfo *foundry_dependency_provider_dup_plugin_info     (FoundryDependencyProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture      *foundry_dependency_provider_list_dependencies   (FoundryDependencyProvider *self,
                                                                 FoundryConfig             *config,
                                                                 FoundryDependency         *parent) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
DexFuture      *foundry_dependency_provider_update_dependencies (FoundryDependencyProvider *self,
                                                                 FoundryConfig             *config,
                                                                 GListModel                *dependencies,
                                                                 int                        pty_fd,
                                                                 DexCancellable            *cancellable) G_GNUC_WARN_UNUSED_RESULT;

G_END_DECLS
