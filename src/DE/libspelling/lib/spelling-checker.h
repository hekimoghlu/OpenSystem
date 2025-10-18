/*
 * spelling-checker.h
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

#include <glib-object.h>

#include "spelling-types.h"
#include "spelling-version-macros.h"

G_BEGIN_DECLS

#define SPELLING_TYPE_CHECKER (spelling_checker_get_type())

SPELLING_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (SpellingChecker, spelling_checker, SPELLING, CHECKER, GObject)

SPELLING_AVAILABLE_IN_ALL
SpellingChecker   *spelling_checker_get_default          (void);
SPELLING_AVAILABLE_IN_ALL
SpellingChecker   *spelling_checker_new                  (SpellingProvider *provider,
                                                          const char       *language);
SPELLING_AVAILABLE_IN_ALL
SpellingProvider  *spelling_checker_get_provider         (SpellingChecker  *self);
SPELLING_AVAILABLE_IN_ALL
const char        *spelling_checker_get_language         (SpellingChecker  *self);
SPELLING_AVAILABLE_IN_ALL
void               spelling_checker_set_language         (SpellingChecker  *self,
                                                          const char       *language);
SPELLING_AVAILABLE_IN_ALL
gboolean           spelling_checker_check_word           (SpellingChecker  *self,
                                                          const char       *word,
                                                          gssize            word_len);
SPELLING_AVAILABLE_IN_ALL
char             **spelling_checker_list_corrections     (SpellingChecker  *self,
                                                          const char       *word);
SPELLING_AVAILABLE_IN_ALL
void               spelling_checker_add_word             (SpellingChecker  *self,
                                                          const char       *word);
SPELLING_AVAILABLE_IN_ALL
void               spelling_checker_ignore_word          (SpellingChecker  *self,
                                                          const char       *word);
SPELLING_AVAILABLE_IN_ALL
const char        *spelling_checker_get_extra_word_chars (SpellingChecker  *self);

G_END_DECLS
