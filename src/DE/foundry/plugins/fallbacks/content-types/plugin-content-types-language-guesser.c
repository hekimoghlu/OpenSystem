/* plugin-content-types-language-guesser.c
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

#include "content-types.h"

#include "plugin-content-types-language-guesser.h"

struct _PluginContentTypesLanguageGuesser
{
  FoundryLanguageGuesser parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginContentTypesLanguageGuesser, plugin_content_types_language_guesser, FOUNDRY_TYPE_LANGUAGE_GUESSER)

static DexFuture *
plugin_content_types_language_guesser_guess (FoundryLanguageGuesser *guesser,
                                             GFile                  *file,
                                             const char             *content_type,
                                             GBytes                 *contents)
{
  if (file != NULL || content_type != NULL)
    {
      g_autofree char *name = file ? g_file_get_basename (file) : NULL;

      for (guint i = 0; i < G_N_ELEMENTS (languages); i++)
        {
          if (name != NULL && languages[i].globs != NULL)
            {
              for (guint j = 0; languages[i].globs[j]; j++)
                {
                  if (g_pattern_match_simple (languages[i].globs[j], name))
                    return dex_future_new_take_string (g_strdup (languages[i].language));
                }
            }

          if (content_type != NULL && languages[i].content_types != NULL)
            {
              for (guint j = 0; languages[i].content_types[j]; j++)
                if (strcmp (content_type, languages[i].content_types[j]) == 0)
                  return dex_future_new_take_string (g_strdup (languages[i].language));
            }
        }
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "No language was found");
}

static char **
plugin_content_types_language_guesser_list_languages (FoundryLanguageGuesser *guesser)
{
  g_autoptr(GStrvBuilder) builder = g_strv_builder_new ();

  for (guint i = 0; i < G_N_ELEMENTS (languages); i++)
    g_strv_builder_add (builder, languages[i].language);

  return g_strv_builder_end (builder);
}

static void
plugin_content_types_language_guesser_class_init (PluginContentTypesLanguageGuesserClass *klass)
{
  FoundryLanguageGuesserClass *language_guesser_class = FOUNDRY_LANGUAGE_GUESSER_CLASS (klass);

  language_guesser_class->guess = plugin_content_types_language_guesser_guess;
  language_guesser_class->list_languages = plugin_content_types_language_guesser_list_languages;
}

static void
plugin_content_types_language_guesser_init (PluginContentTypesLanguageGuesser *self)
{
}
