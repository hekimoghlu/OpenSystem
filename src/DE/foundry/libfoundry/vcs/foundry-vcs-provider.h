/* foundry-vcs-provider.h
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

#include "foundry-contextual.h"
#include "foundry-operation.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_VCS_PROVIDER (foundry_vcs_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryVcsProvider, foundry_vcs_provider, FOUNDRY, VCS_PROVIDER, FoundryContextual)

struct _FoundryVcsProviderClass
{
  FoundryContextualClass parent_class;

  char      *(*dup_name)     (FoundryVcsProvider *self);
  DexFuture *(*load)         (FoundryVcsProvider *self);
  DexFuture *(*unload)       (FoundryVcsProvider *self);
  gboolean   (*supports_uri) (FoundryVcsProvider *self,
                              const char         *uri_string);

  /*< private >*/
  gpointer _reserved[8];
};

FOUNDRY_AVAILABLE_IN_ALL
char     *foundry_vcs_provider_dup_name     (FoundryVcsProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean  foundry_vcs_provider_supports_uri (FoundryVcsProvider *self,
                                             const char         *uri_string);
FOUNDRY_AVAILABLE_IN_ALL
void      foundry_vcs_provider_set_vcs      (FoundryVcsProvider *self,
                                             FoundryVcs         *vcs);

G_END_DECLS
