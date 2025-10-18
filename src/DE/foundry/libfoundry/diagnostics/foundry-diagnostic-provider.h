/* foundry-diagnostic-provider.h
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

#include <libpeas.h>

#include "foundry-contextual.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DIAGNOSTIC_PROVIDER (foundry_diagnostic_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDiagnosticProvider, foundry_diagnostic_provider, FOUNDRY, DIAGNOSTIC_PROVIDER, FoundryContextual)

struct _FoundryDiagnosticProviderClass
{
  FoundryContextualClass parent_class;

  char      *(*dup_name) (FoundryDiagnosticProvider *self);
  DexFuture *(*load)     (FoundryDiagnosticProvider *self);
  DexFuture *(*unload)   (FoundryDiagnosticProvider *self);
  DexFuture *(*diagnose) (FoundryDiagnosticProvider *self,
                          GFile                     *file,
                          GBytes                    *contents,
                          const char                *language);
  DexFuture *(*list_all) (FoundryDiagnosticProvider *self);

  /*< private >*/
  gpointer _reserved[7];
};

FOUNDRY_AVAILABLE_IN_ALL
char           *foundry_diagnostic_provider_dup_name        (FoundryDiagnosticProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
PeasPluginInfo *foundry_diagnostic_provider_dup_plugin_info (FoundryDiagnosticProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture      *foundry_diagnostic_provider_diagnose        (FoundryDiagnosticProvider *self,
                                                             GFile                     *file,
                                                             GBytes                    *contents,
                                                             const char                *language) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_1_1
DexFuture      *foundry_diagnostic_provider_list_all        (FoundryDiagnosticProvider *self) G_GNUC_WARN_UNUSED_RESULT;

G_END_DECLS
