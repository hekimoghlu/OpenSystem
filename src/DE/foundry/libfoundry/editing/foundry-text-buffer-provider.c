/* foundry-text-buffer-provider.c
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

#include "foundry-text-buffer.h"
#include "foundry-text-buffer-provider.h"

G_DEFINE_ABSTRACT_TYPE (FoundryTextBufferProvider, foundry_text_buffer_provider, FOUNDRY_TYPE_CONTEXTUAL)

static void
foundry_text_buffer_provider_class_init (FoundryTextBufferProviderClass *klass)
{
}

static void
foundry_text_buffer_provider_init (FoundryTextBufferProvider *self)
{
}

/**
 * foundry_text_buffer_provider_create_buffer:
 * @self: a [class@Foundry.TextBufferProvider]
 *
 * Returns: (transfer full):
 */
FoundryTextBuffer *
foundry_text_buffer_provider_create_buffer (FoundryTextBufferProvider *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_BUFFER_PROVIDER (self), NULL);

  return FOUNDRY_TEXT_BUFFER_PROVIDER_GET_CLASS (self)->create_buffer (self);
}

/**
 * foundry_text_buffer_provider_load:
 * @self: a [class@Foundry.TextBufferProvider]
 * @operation: (nullable):
 * @encoding: (nullable):
 * @crlf: (nullable):
 *
 * Returns: (transfer full):
 */
DexFuture *
foundry_text_buffer_provider_load (FoundryTextBufferProvider *self,
                                   FoundryTextBuffer         *buffer,
                                   GFile                     *file,
                                   FoundryOperation          *operation,
                                   const char                *encoding,
                                   const char                *crlf)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_BUFFER_PROVIDER (self), NULL);
  g_return_val_if_fail (FOUNDRY_IS_TEXT_BUFFER (buffer), NULL);
  g_return_val_if_fail (G_IS_FILE (file), NULL);
  g_return_val_if_fail (!operation || FOUNDRY_IS_OPERATION (operation), NULL);

  return FOUNDRY_TEXT_BUFFER_PROVIDER_GET_CLASS (self)->load (self, buffer, file, operation, encoding, crlf);
}

/**
 * foundry_text_buffer_provider_save:
 * @self: a [class@Foundry.TextBufferProvider]
 * @operation: (nullable):
 * @encoding: (nullable):
 * @crlf: (nullable):
 *
 * Returns: (transfer full):
 */
DexFuture *
foundry_text_buffer_provider_save (FoundryTextBufferProvider *self,
                                   FoundryTextBuffer         *buffer,
                                   GFile                     *file,
                                   FoundryOperation          *operation,
                                   const char                *encoding,
                                   const char                *crlf)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_BUFFER_PROVIDER (self), NULL);
  g_return_val_if_fail (FOUNDRY_IS_TEXT_BUFFER (buffer), NULL);
  g_return_val_if_fail (G_IS_FILE (file), NULL);
  g_return_val_if_fail (!operation || FOUNDRY_IS_OPERATION (operation), NULL);

  return FOUNDRY_TEXT_BUFFER_PROVIDER_GET_CLASS (self)->save (self, buffer, file, operation, encoding, crlf);
}
