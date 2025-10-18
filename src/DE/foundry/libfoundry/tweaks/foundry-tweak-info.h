/* foundry-tweak-info.h
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

G_BEGIN_DECLS

#ifndef __GI_SCANNER__

typedef enum _FoundryTweakType
{
  FOUNDRY_TWEAK_TYPE_GROUP  = 1,
  FOUNDRY_TWEAK_TYPE_SWITCH = 2,
  FOUNDRY_TWEAK_TYPE_SPIN   = 3,
  FOUNDRY_TWEAK_TYPE_FONT   = 4,
  FOUNDRY_TWEAK_TYPE_COMBO  = 5,
} FoundryTweakType;

typedef enum _FoundryTweakSourceType
{
  FOUNDRY_TWEAK_SOURCE_TYPE_SETTING  = 1,
  FOUNDRY_TWEAK_SOURCE_TYPE_CALLBACK = 2,
} FoundryTweakSourceType;

typedef FoundryInput *(*FoundryTweakCallback) (const FoundryTweakInfo *info,
                                               const char             *path,
                                               FoundryContext         *context);

typedef struct _FoundryTweakSource
{
  FoundryTweakSourceType type;
  union {
    struct {
      const char *schema_id;
      const char *path;
      const char *key;
    } setting;
    struct {
      FoundryTweakCallback callback;
    } callback;
    /*< private >*/
    gpointer _reserved[15];
  };
} FoundryTweakSource;

typedef enum _FoundryTweakInfoFlags
{
  FOUNDRY_TWEAK_INFO_FONT_MONOSPACE = 1 << 0,
} FoundryTweakInfoFlags;

struct _FoundryTweakInfo
{
  FoundryTweakType    type;
  guint               flags;
  const char         *subpath;
  const char         *title;
  const char         *subtitle;
  const char         *icon_name;
  const char         *display_hint;
  const char         *sort_key;
  FoundryTweakSource *source;
  const char         *section;

  /*< private >*/
  gpointer _reserved[7];
};

#endif

G_END_DECLS
