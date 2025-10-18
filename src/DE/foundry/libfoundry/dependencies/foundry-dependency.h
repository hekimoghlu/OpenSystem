/* foundry-dependency.h
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

#include <libdex.h>

#include "foundry-contextual.h"
#include "foundry-dependency-provider.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_DEPENDENCY (foundry_dependency_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryDependency, foundry_dependency, FOUNDRY, DEPENDENCY, FoundryContextual)

struct _FoundryDependencyClass
{
  FoundryContextualClass parent_class;

  FoundryDependencyProvider *(*dup_provider) (FoundryDependency *self);
  char                      *(*dup_name)     (FoundryDependency *self);
  char                      *(*dup_kind)     (FoundryDependency *self);
  char                      *(*dup_location) (FoundryDependency *self);

  /*< private >*/
  gpointer _reserved[12];
};

FOUNDRY_AVAILABLE_IN_ALL
FoundryDependencyProvider *foundry_dependency_dup_provider (FoundryDependency *self);
FOUNDRY_AVAILABLE_IN_ALL
char                      *foundry_dependency_dup_kind     (FoundryDependency *self);
FOUNDRY_AVAILABLE_IN_ALL
char                      *foundry_dependency_dup_name     (FoundryDependency *self);
FOUNDRY_AVAILABLE_IN_ALL
char                      *foundry_dependency_dup_location (FoundryDependency *self);

G_END_DECLS
