/* foundry-source-completion-request.c
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
#include "foundry-source-completion-request-private.h"

struct _FoundrySourceCompletionRequest
{
  FoundryCompletionRequest    parent_instance;
  GtkSourceCompletionContext *context;
};

G_DEFINE_FINAL_TYPE (FoundrySourceCompletionRequest, foundry_source_completion_request, FOUNDRY_TYPE_COMPLETION_REQUEST)

static GFile *
foundry_source_completion_request_dup_file (FoundryCompletionRequest *request)
{
  FoundrySourceCompletionRequest *self = FOUNDRY_SOURCE_COMPLETION_REQUEST (request);
  GtkSourceBuffer *buffer = gtk_source_completion_context_get_buffer (self->context);

  return _foundry_source_buffer_dup_file (FOUNDRY_SOURCE_BUFFER (buffer));
}

static char *
foundry_source_completion_request_dup_language_id (FoundryCompletionRequest *request)
{
  FoundrySourceCompletionRequest *self = FOUNDRY_SOURCE_COMPLETION_REQUEST (request);
  GtkSourceBuffer *buffer = gtk_source_completion_context_get_buffer (self->context);

  return foundry_text_buffer_dup_language_id (FOUNDRY_TEXT_BUFFER (buffer));
}

static void
foundry_source_completion_request_get_bounds (FoundryCompletionRequest *request,
                                              FoundryTextIter          *begin,
                                              FoundryTextIter          *end)
{
  FoundrySourceCompletionRequest *self = FOUNDRY_SOURCE_COMPLETION_REQUEST (request);
  FoundryTextBuffer *buffer;
  GtkTextIter tbegin, tend;

  gtk_source_completion_context_get_bounds (self->context, &tbegin, &tend);

  buffer = FOUNDRY_TEXT_BUFFER (gtk_source_completion_context_get_buffer (self->context));

  foundry_text_buffer_get_start_iter (buffer, begin);
  foundry_text_iter_move_to_line_and_offset (begin,
                                             gtk_text_iter_get_line (&tbegin),
                                             gtk_text_iter_get_line_offset (&tbegin));

  foundry_text_buffer_get_start_iter (buffer, end);
  foundry_text_iter_move_to_line_and_offset (end,
                                             gtk_text_iter_get_line (&tend),
                                             gtk_text_iter_get_line_offset (&tend));
}

static FoundryCompletionActivation
foundry_source_completion_request_get_activation (FoundryCompletionRequest *request)
{
  FoundrySourceCompletionRequest *self = FOUNDRY_SOURCE_COMPLETION_REQUEST (request);

  switch (gtk_source_completion_context_get_activation (self->context))
    {
    default:
    case GTK_SOURCE_COMPLETION_ACTIVATION_NONE:
      return FOUNDRY_COMPLETION_ACTIVATION_NONE;

    case GTK_SOURCE_COMPLETION_ACTIVATION_USER_REQUESTED:
      return FOUNDRY_COMPLETION_ACTIVATION_USER_REQUESTED;

    case GTK_SOURCE_COMPLETION_ACTIVATION_INTERACTIVE:
      return FOUNDRY_COMPLETION_ACTIVATION_INTERACTIVE;
    }
}

static char *
foundry_source_completion_request_dup_word (FoundryCompletionRequest *request)
{
  FoundrySourceCompletionRequest *self = FOUNDRY_SOURCE_COMPLETION_REQUEST (request);

  return gtk_source_completion_context_get_word (self->context);
}

static void
foundry_source_completion_request_dispose (GObject *object)
{
  FoundrySourceCompletionRequest *self = (FoundrySourceCompletionRequest *)object;

  g_clear_object (&self->context);

  G_OBJECT_CLASS (foundry_source_completion_request_parent_class)->dispose (object);
}

static void
foundry_source_completion_request_class_init (FoundrySourceCompletionRequestClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryCompletionRequestClass *request_class = FOUNDRY_COMPLETION_REQUEST_CLASS (klass);

  object_class->dispose = foundry_source_completion_request_dispose;

  request_class->dup_file = foundry_source_completion_request_dup_file;
  request_class->dup_language_id = foundry_source_completion_request_dup_language_id;
  request_class->get_bounds = foundry_source_completion_request_get_bounds;
  request_class->get_activation = foundry_source_completion_request_get_activation;
  request_class->dup_word = foundry_source_completion_request_dup_word;
}

static void
foundry_source_completion_request_init (FoundrySourceCompletionRequest *self)
{
}

FoundryCompletionRequest *
foundry_source_completion_request_new (GtkSourceCompletionContext *context)
{
  FoundrySourceCompletionRequest *self;

  g_return_val_if_fail (GTK_SOURCE_IS_COMPLETION_CONTEXT (context), NULL);

  self = g_object_new (FOUNDRY_TYPE_SOURCE_COMPLETION_REQUEST, NULL);
  self->context = g_object_ref (context);

  return FOUNDRY_COMPLETION_REQUEST (self);
}
