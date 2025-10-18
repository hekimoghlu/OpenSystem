/* foundry-vcs-signature.h
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

#include <glib-object.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_VCS_SIGNATURE (foundry_vcs_signature_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryVcsSignature, foundry_vcs_signature, FOUNDRY, VCS_SIGNATURE, GObject)

struct _FoundryVcsSignatureClass
{
  GObjectClass parent_class;

  char      *(*dup_name)  (FoundryVcsSignature *self);
  char      *(*dup_email) (FoundryVcsSignature *self);
  GDateTime *(*dup_when)  (FoundryVcsSignature *self);

  /*< private >*/
  gpointer _reserved[12];
};

FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_vcs_signature_dup_name  (FoundryVcsSignature *self);
FOUNDRY_AVAILABLE_IN_ALL
char      *foundry_vcs_signature_dup_email (FoundryVcsSignature *self);
FOUNDRY_AVAILABLE_IN_ALL
GDateTime *foundry_vcs_signature_dup_when  (FoundryVcsSignature *self);

G_END_DECLS
