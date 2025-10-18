/* foundry-source-indenter.c
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

#include "foundry-source-indenter-private.h"
#include "foundry-source-buffer-private.h"
#include "foundry-source-view.h"

struct _FoundrySourceIndenter
{
  GObject                 parent_instance;
  FoundryOnTypeFormatter *formatter;
};

static gboolean
foundry_source_indenter_is_trigger (GtkSourceIndenter *indenter,
                                    GtkSourceView     *view,
                                    const GtkTextIter *location,
                                    GdkModifierType    state,
                                    guint              keyval)
{
  FoundrySourceIndenter *self = (FoundrySourceIndenter *)indenter;
  g_autoptr(FoundryTextDocument) document = NULL;
  GtkTextBuffer *buffer;
  FoundryTextIter real;
  FoundryModifierType type = 0;

  g_assert (FOUNDRY_IS_SOURCE_INDENTER (self));
  g_assert (FOUNDRY_IS_SOURCE_VIEW (view));

  document = foundry_source_view_dup_document (FOUNDRY_SOURCE_VIEW (view));
  buffer = gtk_text_view_get_buffer (GTK_TEXT_VIEW (view));

  g_assert (FOUNDRY_IS_SOURCE_BUFFER (buffer));

  _foundry_source_buffer_init_iter (FOUNDRY_SOURCE_BUFFER (buffer), &real, location);

  if (state & GDK_CONTROL_MASK)
    type |= FOUNDRY_MODIFIER_CONTROL;

  if (state & GDK_ALT_MASK)
    type |= FOUNDRY_MODIFIER_ALT;

  if (state & GDK_SHIFT_MASK)
    type |= FOUNDRY_MODIFIER_SHIFT;

  if (state & GDK_SUPER_MASK)
    type |= FOUNDRY_MODIFIER_SUPER;

#ifdef __apple__
  if (state & GDK_META_MASK)
    type |= FOUNDRY_MODIFIER_COMMAND;
#endif

  return foundry_on_type_formatter_is_trigger (self->formatter, document, &real, type, keyval);
}

static void
foundry_source_indenter_indent (GtkSourceIndenter *indenter,
                                GtkSourceView     *view,
                                GtkTextIter       *iter)
{
  FoundrySourceIndenter *self = (FoundrySourceIndenter *)indenter;
  g_autoptr(FoundryTextDocument) document = NULL;
  GtkTextBuffer *buffer;
  FoundryTextIter real;

  g_assert (FOUNDRY_IS_SOURCE_INDENTER (self));
  g_assert (FOUNDRY_IS_SOURCE_VIEW (view));

  document = foundry_source_view_dup_document (FOUNDRY_SOURCE_VIEW (view));
  buffer = gtk_text_view_get_buffer (GTK_TEXT_VIEW (view));

  g_assert (FOUNDRY_IS_SOURCE_BUFFER (buffer));

  _foundry_source_buffer_init_iter (FOUNDRY_SOURCE_BUFFER (buffer), &real, iter);

  foundry_on_type_formatter_indent (self->formatter, document, &real);

  gtk_text_iter_set_line (iter, foundry_text_iter_get_line (&real));
  gtk_text_iter_set_line_offset (iter, foundry_text_iter_get_line_offset (&real));
}

static void
indenter_iface_init (GtkSourceIndenterInterface *iface)
{
  iface->is_trigger = foundry_source_indenter_is_trigger;
  iface->indent = foundry_source_indenter_indent;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundrySourceIndenter, foundry_source_indenter, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (GTK_SOURCE_TYPE_INDENTER, indenter_iface_init))

static void
foundry_source_indenter_dispose (GObject *object)
{
  FoundrySourceIndenter *self = (FoundrySourceIndenter *)object;

  g_clear_object (&self->formatter);

  G_OBJECT_CLASS (foundry_source_indenter_parent_class)->dispose (object);
}

static void
foundry_source_indenter_class_init (FoundrySourceIndenterClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_source_indenter_dispose;
}

static void
foundry_source_indenter_init (FoundrySourceIndenter *self)
{
}

GtkSourceIndenter *
foundry_source_indenter_new (FoundryOnTypeFormatter *formatter)
{
  FoundrySourceIndenter *self;

  g_return_val_if_fail (FOUNDRY_IS_ON_TYPE_FORMATTER (formatter), NULL);

  self = g_object_new (FOUNDRY_TYPE_SOURCE_INDENTER, NULL);
  self->formatter = g_object_ref (formatter);

  return GTK_SOURCE_INDENTER (self);
}
