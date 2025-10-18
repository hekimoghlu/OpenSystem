/* foundry-source-language-guesser.c
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

#include <gtksourceview/gtksource.h>

#include "foundry-source-language-guesser.h"

struct _FoundrySourceLanguageGuesser
{
  FoundryLanguageGuesser parent_instance;
};

G_DEFINE_FINAL_TYPE (FoundrySourceLanguageGuesser, foundry_source_language_guesser, FOUNDRY_TYPE_LANGUAGE_GUESSER)

static DexFuture *
foundry_source_language_guesser_guess (FoundryLanguageGuesser *guesser,
                                       GFile                  *file,
                                       const char             *content_type,
                                       GBytes                 *contents)
{
  GtkSourceLanguageManager *manager;
  GtkSourceLanguage *language;
  g_autofree char *path = NULL;

  g_assert (FOUNDRY_IS_SOURCE_LANGUAGE_GUESSER (guesser));
  g_assert (G_IS_FILE (file));

  manager = gtk_source_language_manager_get_default ();

  if (file != NULL)
    path = g_file_get_path (file);

  if ((language = gtk_source_language_manager_guess_language (manager, path, content_type)))
    return dex_future_new_take_string (g_strdup (gtk_source_language_get_id (language)));

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Failed to locate language");
}

static char **
foundry_source_language_guesser_list_languages (FoundryLanguageGuesser *guesser)
{
  GtkSourceLanguageManager *manager = gtk_source_language_manager_get_default ();

  return g_strdupv ((char **)gtk_source_language_manager_get_language_ids (manager));
}

static void
foundry_source_language_guesser_class_init (FoundrySourceLanguageGuesserClass *klass)
{
  FoundryLanguageGuesserClass *guesser_class = FOUNDRY_LANGUAGE_GUESSER_CLASS (klass);

  guesser_class->guess = foundry_source_language_guesser_guess;
  guesser_class->list_languages = foundry_source_language_guesser_list_languages;
}

static void
foundry_source_language_guesser_init (FoundrySourceLanguageGuesser *self)
{
}
