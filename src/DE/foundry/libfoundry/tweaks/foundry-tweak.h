/* foundry-tweak.h
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

#include <gio/gio.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TWEAK (foundry_tweak_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryTweak, foundry_tweak, FOUNDRY, TWEAK, GObject)

struct _FoundryTweakClass
{
  GObjectClass parent_class;

  char         *(*dup_path)         (FoundryTweak   *self);
  char         *(*dup_title)        (FoundryTweak   *self);
  char         *(*dup_subtitle)     (FoundryTweak   *self);
  char         *(*dup_display_hint) (FoundryTweak   *self);
  char         *(*dup_section)      (FoundryTweak   *self);
  char         *(*dup_sort_key)     (FoundryTweak   *self);
  GIcon        *(*dup_icon)         (FoundryTweak   *self);
  FoundryInput *(*create_input)     (FoundryTweak   *self,
                                     FoundryContext *context);

  /*< private >*/
  gpointer _reserved[15];
};

FOUNDRY_AVAILABLE_IN_ALL
FoundryInput *foundry_tweak_create_input     (FoundryTweak   *self,
                                              FoundryContext *context);
FOUNDRY_AVAILABLE_IN_ALL
char         *foundry_tweak_dup_path         (FoundryTweak   *self);
FOUNDRY_AVAILABLE_IN_ALL
char         *foundry_tweak_dup_title        (FoundryTweak   *self);
FOUNDRY_AVAILABLE_IN_ALL
char         *foundry_tweak_dup_subtitle     (FoundryTweak   *self);
FOUNDRY_AVAILABLE_IN_ALL
char         *foundry_tweak_dup_display_hint (FoundryTweak   *self);
FOUNDRY_AVAILABLE_IN_ALL
char         *foundry_tweak_dup_sort_key     (FoundryTweak   *self);
FOUNDRY_AVAILABLE_IN_ALL
char         *foundry_tweak_dup_section      (FoundryTweak   *self);
FOUNDRY_AVAILABLE_IN_ALL
GIcon        *foundry_tweak_dup_icon         (FoundryTweak   *self);

G_END_DECLS
