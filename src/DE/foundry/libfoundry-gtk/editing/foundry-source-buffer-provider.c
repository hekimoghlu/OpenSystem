/* foundry-source-buffer-provider.c
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

#include "foundry-source-buffer-private.h"
#include "foundry-source-buffer-provider-private.h"
#include "foundry-sourceview.h"

#define METADATA_CURSOR   "metadata::foundry-cursor"
#define METADATA_SYNTAX   "metadata::foundry-syntax"

struct _FoundrySourceBufferProvider
{
  FoundryTextBufferProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (FoundrySourceBufferProvider, foundry_source_buffer_provider, FOUNDRY_TYPE_TEXT_BUFFER_PROVIDER)

static FoundryTextBuffer *
foundry_source_buffer_provider_create_buffer (FoundryTextBufferProvider *provider)
{
  g_autoptr(FoundryContext) context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (provider));
  FoundrySourceBuffer *buffer = _foundry_source_buffer_new (context, NULL);

  return FOUNDRY_TEXT_BUFFER (buffer);
}

static DexFuture *
foundry_source_buffer_provider_load_fiber (FoundryContext    *context,
                                           FoundryTextBuffer *buffer,
                                           GFile             *location,
                                           FoundryOperation  *operation,
                                           const char        *charset,
                                           const char        *crlf)
{
  g_autoptr(GtkSourceFileLoader) loader = NULL;
  g_autoptr(FoundryFileManager) file_manager = NULL;
  g_autoptr(GtkSourceFile) file = NULL;
  g_autoptr(GFileInfo) file_info = NULL;
  g_autoptr(GBytes) sniff = NULL;
  g_autoptr(GError) error = NULL;
  GtkTextIter begin, end;
  const char *language;
  char *text;

  g_assert (FOUNDRY_IS_CONTEXT (context));
  g_assert (FOUNDRY_IS_SOURCE_BUFFER (buffer));
  g_assert (G_IS_FILE (location));
  g_assert (!operation || FOUNDRY_IS_OPERATION (operation));

  file_manager = foundry_context_dup_file_manager (context);

  file = gtk_source_file_new ();
  gtk_source_file_set_location (file, location);

  loader = gtk_source_file_loader_new (GTK_SOURCE_BUFFER (buffer), file);

  if (charset != NULL)
    {
      const GtkSourceEncoding *encoding = gtk_source_encoding_get_from_charset (charset);

      if (encoding != NULL)
        {
          GSList candidate = { .next = NULL, .data = (gpointer)encoding };
          gtk_source_file_loader_set_candidate_encodings (loader, &candidate);
        }
    }

  if (!dex_await (gtk_source_file_loader_load (loader, G_PRIORITY_DEFAULT, operation), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  gtk_text_buffer_get_bounds (GTK_TEXT_BUFFER (buffer), &begin, &end);

  /* Move cursor to the start */
  gtk_text_buffer_select_range (GTK_TEXT_BUFFER (buffer), &begin, &begin);

  /* Grab the first 1KB of data for sniffing content-type */
  if (gtk_text_iter_get_offset (&end) > 1024)
    gtk_text_iter_set_offset (&end, 1024);
  text = gtk_text_iter_get_slice (&begin, &end);
  sniff = g_bytes_new_take (text, strlen (text));

  /* Sniff syntax language from file and buffer contents */
  if ((language = dex_await_string (foundry_file_manager_guess_language (file_manager, location, NULL, sniff), NULL)))
    {
      GtkSourceLanguageManager *lm = gtk_source_language_manager_get_default ();
      GtkSourceLanguage *l = gtk_source_language_manager_get_language (lm, language);

      if (l != NULL)
        gtk_source_buffer_set_language (GTK_SOURCE_BUFFER (buffer), l);
    }

  if ((file_info = dex_await_object (foundry_file_manager_read_metadata (file_manager, location, "metadata::*"), NULL)))
    {
      const char *cursor;
      const char *override_syntax;

      if ((cursor = g_file_info_get_attribute_string (file_info, METADATA_CURSOR)))
        {
          GtkTextIter iter;
          guint line = 0;
          guint line_offset = 0;

          if (sscanf (cursor, "%u:%u", &line, &line_offset) == 2)
            {
              gtk_text_buffer_get_iter_at_line_offset (GTK_TEXT_BUFFER (buffer),
                                                       &iter, line, line_offset);
              gtk_text_buffer_select_range (GTK_TEXT_BUFFER (buffer), &iter, &iter);
            }
        }

      if ((override_syntax = g_file_info_get_attribute_string (file_info, METADATA_SYNTAX)))
        foundry_source_buffer_set_override_syntax (FOUNDRY_SOURCE_BUFFER (buffer), override_syntax);
    }

  _foundry_source_buffer_set_file (FOUNDRY_SOURCE_BUFFER (buffer), location);

  return dex_future_new_true ();
}

static DexFuture *
foundry_source_buffer_provider_load (FoundryTextBufferProvider *provider,
                                     FoundryTextBuffer         *buffer,
                                     GFile                     *file,
                                     FoundryOperation          *operation,
                                     const char                *encoding,
                                     const char                *crlf)
{
  g_autoptr(FoundryContext) context = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_SOURCE_BUFFER_PROVIDER (provider));
  dex_return_error_if_fail (FOUNDRY_IS_SOURCE_BUFFER (buffer));
  dex_return_error_if_fail (G_IS_FILE (file));
  dex_return_error_if_fail (FOUNDRY_IS_OPERATION (operation));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (provider));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_source_buffer_provider_load_fiber),
                                  6,
                                  FOUNDRY_TYPE_CONTEXT, context,
                                  FOUNDRY_TYPE_TEXT_BUFFER, buffer,
                                  G_TYPE_FILE, file,
                                  FOUNDRY_TYPE_OPERATION, operation,
                                  G_TYPE_STRING, encoding,
                                  G_TYPE_STRING, crlf);
}

static DexFuture *
foundry_source_buffer_provider_save_fiber (FoundryContext    *context,
                                           FoundryTextBuffer *buffer,
                                           GFile             *location,
                                           FoundryOperation  *operation,
                                           const char        *charset,
                                           const char        *crlf)
{
  g_autoptr(GtkSourceFileSaver) saver = NULL;
  g_autoptr(FoundryFileManager) file_manager = NULL;
  g_autoptr(GtkSourceFile) file = NULL;
  g_autoptr(GFileInfo) file_info = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *cursor_value = NULL;
  g_autofree char *override_syntax = NULL;
  GtkTextIter cursor;

  g_assert (FOUNDRY_IS_SOURCE_BUFFER (buffer));
  g_assert (G_IS_FILE (location));
  g_assert (!operation || FOUNDRY_IS_OPERATION (operation));

  file_manager = foundry_context_dup_file_manager (context);

  file = gtk_source_file_new ();
  gtk_source_file_set_location (file, location);

  _foundry_source_buffer_set_file (FOUNDRY_SOURCE_BUFFER (buffer), location);

  saver = gtk_source_file_saver_new (GTK_SOURCE_BUFFER (buffer), file);
  gtk_source_file_saver_set_flags (saver,
                                   (GTK_SOURCE_FILE_SAVER_FLAGS_IGNORE_MODIFICATION_TIME |
                                    GTK_SOURCE_FILE_SAVER_FLAGS_IGNORE_INVALID_CHARS));

  if (crlf != NULL)
    {
      if (strcmp (crlf, "\n") == 0)
        gtk_source_file_saver_set_newline_type (saver, GTK_SOURCE_NEWLINE_TYPE_LF);
      else if (strcmp (crlf, "\r") == 0)
        gtk_source_file_saver_set_newline_type (saver, GTK_SOURCE_NEWLINE_TYPE_CR);
      else if (strcmp (crlf, "\r\n") == 0)
        gtk_source_file_saver_set_newline_type (saver, GTK_SOURCE_NEWLINE_TYPE_CR_LF);
    }

  if (charset != NULL)
    {
      const GtkSourceEncoding *encoding = gtk_source_encoding_get_from_charset (charset);

      if (encoding != NULL)
        gtk_source_file_saver_set_encoding (saver, encoding);
    }

  if (!dex_await (gtk_source_file_saver_save (saver, G_PRIORITY_DEFAULT, operation), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  gtk_text_buffer_get_iter_at_mark (GTK_TEXT_BUFFER (buffer),
                                    &cursor,
                                    gtk_text_buffer_get_insert (GTK_TEXT_BUFFER (buffer)));
  cursor_value = g_strdup_printf ("%u:%u",
                                  gtk_text_iter_get_line (&cursor),
                                  gtk_text_iter_get_line_offset (&cursor));

  file_info = g_file_info_new ();
  g_file_info_set_attribute_string (file_info, METADATA_CURSOR, cursor_value);

  if ((override_syntax = foundry_source_buffer_dup_override_syntax (FOUNDRY_SOURCE_BUFFER (buffer))))
    g_file_info_set_attribute_string (file_info, METADATA_SYNTAX, override_syntax);

  if (!dex_await (foundry_file_manager_write_metadata (file_manager, location, file_info), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

static DexFuture *
foundry_source_buffer_provider_save (FoundryTextBufferProvider *provider,
                                     FoundryTextBuffer         *buffer,
                                     GFile                     *file,
                                     FoundryOperation          *operation,
                                     const char                *encoding,
                                     const char                *crlf)
{
  g_autoptr(FoundryContext) context = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_SOURCE_BUFFER_PROVIDER (provider));
  dex_return_error_if_fail (FOUNDRY_IS_SOURCE_BUFFER (buffer));
  dex_return_error_if_fail (G_IS_FILE (file));
  dex_return_error_if_fail (FOUNDRY_IS_OPERATION (operation));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (provider));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_source_buffer_provider_save_fiber),
                                  5,
                                  FOUNDRY_TYPE_CONTEXT, context,
                                  FOUNDRY_TYPE_TEXT_BUFFER, buffer,
                                  G_TYPE_FILE, file,
                                  FOUNDRY_TYPE_OPERATION, operation,
                                  G_TYPE_STRING, encoding,
                                  G_TYPE_STRING, crlf);
}

static void
foundry_source_buffer_provider_class_init (FoundrySourceBufferProviderClass *klass)
{
  FoundryTextBufferProviderClass *text_buffer_provider_class = FOUNDRY_TEXT_BUFFER_PROVIDER_CLASS (klass);

  text_buffer_provider_class->create_buffer = foundry_source_buffer_provider_create_buffer;
  text_buffer_provider_class->load = foundry_source_buffer_provider_load;
  text_buffer_provider_class->save = foundry_source_buffer_provider_save;
}

static void
foundry_source_buffer_provider_init (FoundrySourceBufferProvider *self)
{
}
