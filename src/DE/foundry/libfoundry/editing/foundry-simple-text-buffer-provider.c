/* foundry-simple-text-buffer-provider.c
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

#include "foundry-file-manager.h"
#include "foundry-simple-text-buffer.h"
#include "foundry-simple-text-buffer-provider.h"
#include "foundry-util.h"

struct _FoundrySimpleTextBufferProvider
{
  FoundryTextBufferProvider parent_instance;
};

G_DEFINE_FINAL_TYPE (FoundrySimpleTextBufferProvider, foundry_simple_text_buffer_provider, FOUNDRY_TYPE_TEXT_BUFFER_PROVIDER)

static FoundryTextBuffer *
foundry_simple_text_buffer_provider_create_buffer (FoundryTextBufferProvider *provider)
{
  g_autoptr(FoundryContext) context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (provider));

  return g_object_new (FOUNDRY_TYPE_SIMPLE_TEXT_BUFFER,
                       "context", context,
                       NULL);
}

static DexFuture *
foundry_simple_text_buffer_provider_save_fiber (GFile  *file,
                                                GBytes *contents)
{
  g_autoptr(GOutputStream) stream = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (contents != NULL);
  g_assert (G_IS_FILE (file));

  if (!(stream = dex_await_object (dex_file_replace (file, NULL, FALSE, 0, 0), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!dex_await (dex_output_stream_write_bytes (stream, contents, 0), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_true ();
}

static DexFuture *
foundry_simple_text_buffer_provider_save (FoundryTextBufferProvider *provider,
                                          FoundryTextBuffer         *buffer,
                                          GFile                     *file,
                                          FoundryOperation          *operation,
                                          const char                *encoding,
                                          const char                *crlf)
{
  g_autoptr(GBytes) contents = NULL;

  g_assert (FOUNDRY_IS_SIMPLE_TEXT_BUFFER_PROVIDER (provider));
  g_assert (G_IS_FILE (file));

  contents = foundry_text_buffer_dup_contents (buffer);

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_simple_text_buffer_provider_save_fiber),
                                  2,
                                  G_TYPE_FILE, file,
                                  G_TYPE_BYTES, contents);
}

static DexFuture *
foundry_simple_text_buffer_provider_load_fiber (FoundryContext          *context,
                                                FoundrySimpleTextBuffer *buffer,
                                                GFile                   *file)
{
  g_autoptr(FoundryFileManager) file_manager = NULL;
  g_autoptr(GBytes) bytes = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *language_id = NULL;
  const guint8 *data;
  gsize len;

  g_assert (FOUNDRY_IS_SIMPLE_TEXT_BUFFER (buffer));
  g_assert (G_IS_FILE (file));

  if (!(bytes = dex_await_boxed (dex_file_load_contents_bytes (file), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  data = g_bytes_get_data (bytes, &len);

  if (!g_utf8_validate_len ((const char *)data, len, NULL))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_INVALID_DATA,
                                  "Data is not UTF-8");

  file_manager = foundry_context_dup_file_manager (context);

  foundry_simple_text_buffer_set_text (buffer, (const char *)data, len);

  if ((language_id = dex_await_string (foundry_file_manager_guess_language (file_manager, file, NULL, bytes), NULL)))
    foundry_simple_text_buffer_set_language_id (buffer, language_id);

  return dex_future_new_true ();
}

static DexFuture *
foundry_simple_text_buffer_provider_load (FoundryTextBufferProvider *provider,
                                          FoundryTextBuffer         *buffer,
                                          GFile                     *file,
                                          FoundryOperation          *operation,
                                          const char                *encoding,
                                          const char                *crlf)
{
  g_autoptr(FoundryContext) context = NULL;

  dex_return_error_if_fail (FOUNDRY_IS_SIMPLE_TEXT_BUFFER_PROVIDER (provider));
  dex_return_error_if_fail (FOUNDRY_IS_SIMPLE_TEXT_BUFFER (buffer));
  dex_return_error_if_fail (G_IS_FILE (file));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (provider));

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_simple_text_buffer_provider_load_fiber),
                                  3,
                                  FOUNDRY_TYPE_CONTEXT, context,
                                  FOUNDRY_TYPE_SIMPLE_TEXT_BUFFER, buffer,
                                  G_TYPE_FILE, file);
}


static void
foundry_simple_text_buffer_provider_class_init (FoundrySimpleTextBufferProviderClass *klass)
{
  FoundryTextBufferProviderClass *text_buffer_provider_class = FOUNDRY_TEXT_BUFFER_PROVIDER_CLASS (klass);

  text_buffer_provider_class->create_buffer = foundry_simple_text_buffer_provider_create_buffer;
  text_buffer_provider_class->load = foundry_simple_text_buffer_provider_load;
  text_buffer_provider_class->save = foundry_simple_text_buffer_provider_save;
}

static void
foundry_simple_text_buffer_provider_init (FoundrySimpleTextBufferProvider *self)
{
}

FoundryTextBufferProvider *
foundry_simple_text_buffer_provider_new (FoundryContext *context)
{
  g_return_val_if_fail (FOUNDRY_IS_CONTEXT (context), NULL);

  return g_object_new (FOUNDRY_TYPE_SIMPLE_TEXT_BUFFER_PROVIDER,
                       "context", context,
                       NULL);
}
