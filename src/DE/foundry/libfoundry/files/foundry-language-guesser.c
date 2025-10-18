/* foundry-language-guesser.c
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

#include "config.h"

#include "foundry-language-guesser.h"

G_DEFINE_ABSTRACT_TYPE (FoundryLanguageGuesser, foundry_language_guesser, FOUNDRY_TYPE_CONTEXTUAL)

static void
foundry_language_guesser_class_init (FoundryLanguageGuesserClass *klass)
{
}

static void
foundry_language_guesser_init (FoundryLanguageGuesser *self)
{
}

/**
 * foundry_language_guesser_guess:
 * @self: a [class@Foundry.LanguageGuesser]
 * @file: (nullable): a [iface@Gio.File] or %NULL
 * @content_type: (nullable): a content-type as a string or %NULL
 * @contents: (nullable): a [struct@GLib.Bytes] of file contents or %NULL
 *
 * Guess the language for a file, content_type, or contents.
 *
 * One of @file, @content_type, or @contents must be set.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a string containing the source code language or rejects with
 *   a new #GError.
 */
DexFuture *
foundry_language_guesser_guess (FoundryLanguageGuesser *self,
                                GFile                  *file,
                                const char             *content_type,
                                GBytes                 *contents)
{
  dex_return_error_if_fail (FOUNDRY_IS_LANGUAGE_GUESSER (self));
  dex_return_error_if_fail (!file || G_IS_FILE (file));
  dex_return_error_if_fail (file || content_type || contents);

  return FOUNDRY_LANGUAGE_GUESSER_GET_CLASS (self)->guess (self, file, content_type, contents);
}

/**
 * foundry_language_guesser_list_languages:
 * @self: a [class@Foundry.LanguageGuesser]
 *
 * Gets a list of known languages by their language identifier.
 *
 * Returns: (transfer full):
 */
char **
foundry_language_guesser_list_languages (FoundryLanguageGuesser *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LANGUAGE_GUESSER (self), NULL);

  if (FOUNDRY_LANGUAGE_GUESSER_GET_CLASS (self)->list_languages)
    return FOUNDRY_LANGUAGE_GUESSER_GET_CLASS (self)->list_languages (self);

  return g_new0 (char *, 1);
}
