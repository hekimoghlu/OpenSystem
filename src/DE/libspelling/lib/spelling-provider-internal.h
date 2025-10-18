/*
 * spelling-provider-internal.h
 *
 * Copyright 2021-2023 Christian Hergert <chergert@redhat.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "spelling-provider.h"

G_BEGIN_DECLS

#define SPELLING_PROVIDER_GET_CLASS(obj) G_TYPE_INSTANCE_GET_CLASS(obj, SPELLING_TYPE_PROVIDER, SpellingProviderClass)

struct _SpellingProvider
{
  GObject parent_instance;
  char *display_name;
};

struct _SpellingProviderClass
{
  GObjectClass parent_class;

  GListModel         *(*list_languages)    (SpellingProvider *self);
  gboolean            (*supports_language) (SpellingProvider *self,
                                            const char       *language);
  SpellingDictionary *(*load_dictionary)   (SpellingProvider *self,
                                            const char       *language);
  const char         *(*get_default_code)  (SpellingProvider *self);
};

G_END_DECLS
