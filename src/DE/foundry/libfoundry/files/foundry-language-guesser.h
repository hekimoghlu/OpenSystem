/* foundry-language-guesser.h
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

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LANGUAGE_GUESSER (foundry_language_guesser_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryLanguageGuesser, foundry_language_guesser, FOUNDRY, LANGUAGE_GUESSER, FoundryContextual)

struct _FoundryLanguageGuesserClass
{
  FoundryContextualClass parent_class;

  DexFuture *(*guess)           (FoundryLanguageGuesser *self,
                                 GFile                  *file,
                                 const char             *content_type,
                                 GBytes                 *contents);
  char      **(*list_languages) (FoundryLanguageGuesser *self);

  /*< private >*/
  gpointer _reserved[6];
};

FOUNDRY_AVAILABLE_IN_ALL
DexFuture  *foundry_language_guesser_guess          (FoundryLanguageGuesser *self,
                                                     GFile                  *file,
                                                     const char             *content_type,
                                                     GBytes                 *contents);
FOUNDRY_AVAILABLE_IN_ALL
char      **foundry_language_guesser_list_languages (FoundryLanguageGuesser *self);

G_END_DECLS
