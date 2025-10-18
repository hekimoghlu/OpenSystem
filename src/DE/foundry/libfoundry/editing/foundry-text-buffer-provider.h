/* foundry-text-buffer-provider.h
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
#include "foundry-operation.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_TEXT_BUFFER_PROVIDER (foundry_text_buffer_provider_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryTextBufferProvider, foundry_text_buffer_provider, FOUNDRY, TEXT_BUFFER_PROVIDER, FoundryContextual)

struct _FoundryTextBufferProviderClass
{
  FoundryContextualClass parent_class;

  FoundryTextBuffer *(*create_buffer) (FoundryTextBufferProvider *self);
  DexFuture         *(*load)          (FoundryTextBufferProvider *self,
                                       FoundryTextBuffer         *buffer,
                                       GFile                     *file,
                                       FoundryOperation          *operation,
                                       const char                *encoding,
                                       const char                *crlf);
  DexFuture         *(*save)          (FoundryTextBufferProvider *self,
                                       FoundryTextBuffer         *buffer,
                                       GFile                     *file,
                                       FoundryOperation          *operation,
                                       const char                *encoding,
                                       const char                *crlf);

  /*< private >*/
  gpointer _reserved[5];
};

FOUNDRY_AVAILABLE_IN_ALL
FoundryTextBuffer *foundry_text_buffer_provider_create_buffer (FoundryTextBufferProvider *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture         *foundry_text_buffer_provider_load          (FoundryTextBufferProvider *self,
                                                               FoundryTextBuffer         *buffer,
                                                               GFile                     *file,
                                                               FoundryOperation          *operation,
                                                               const char                *encoding,
                                                               const char                *crlf);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture         *foundry_text_buffer_provider_save          (FoundryTextBufferProvider *self,
                                                               FoundryTextBuffer         *buffer,
                                                               GFile                     *file,
                                                               FoundryOperation          *operation,
                                                               const char                *encoding,
                                                               const char                *crlf);

G_END_DECLS
