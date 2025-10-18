/*
 * spelling-provider.h
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

#if !defined(LIBSPELLING_INSIDE) && !defined(LIBSPELLING_COMPILATION)
# error "Only <libspelling.h> can be included directly."
#endif

#include <gio/gio.h>

#include "spelling-types.h"
#include "spelling-version-macros.h"

G_BEGIN_DECLS

#define SPELLING_TYPE_PROVIDER         (spelling_provider_get_type())
#define SPELLING_IS_PROVIDER(obj)      (G_TYPE_CHECK_INSTANCE_TYPE(obj, SPELLING_TYPE_PROVIDER))
#define SPELLING_PROVIDER(obj)         (G_TYPE_CHECK_INSTANCE_CAST(obj, SPELLING_TYPE_PROVIDER, SpellingProvider))
#define SPELLING_PROVIDER_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST(klass, SPELLING_TYPE_PROVIDER, SpellingProviderClass))

typedef struct _SpellingProvider SpellingProvider;
typedef struct _SpellingProviderClass SpellingProviderClass;

SPELLING_AVAILABLE_IN_ALL
GType               spelling_provider_get_type          (void) G_GNUC_CONST;
SPELLING_AVAILABLE_IN_ALL
SpellingProvider   *spelling_provider_get_default       (void);
SPELLING_AVAILABLE_IN_ALL
const char         *spelling_provider_get_default_code  (SpellingProvider *self);
SPELLING_AVAILABLE_IN_ALL
const char         *spelling_provider_get_display_name  (SpellingProvider *self);
SPELLING_AVAILABLE_IN_ALL
gboolean            spelling_provider_supports_language (SpellingProvider *self,
                                                         const char       *language);
SPELLING_AVAILABLE_IN_ALL
GListModel         *spelling_provider_list_languages    (SpellingProvider *self);
SPELLING_AVAILABLE_IN_ALL
SpellingDictionary *spelling_provider_load_dictionary   (SpellingProvider *self,
                                                         const char       *language);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (SpellingProvider, g_object_unref)

G_END_DECLS
