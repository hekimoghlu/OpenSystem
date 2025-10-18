/* foundry-text-edit.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include "foundry-text-edit.h"

struct _FoundryTextEdit
{
  GObject parent_instance;
  GFile *file;
  char *replacement;
  guint begin_line;
  int begin_line_offset;
  guint end_line;
  int end_line_offset;
};

G_DEFINE_FINAL_TYPE (FoundryTextEdit, foundry_text_edit, G_TYPE_OBJECT)

static void
foundry_text_edit_finalize (GObject *object)
{
  FoundryTextEdit *self = (FoundryTextEdit *)object;

  g_clear_object (&self->file);
  g_clear_pointer (&self->replacement, g_free);

  G_OBJECT_CLASS (foundry_text_edit_parent_class)->finalize (object);
}

static void
foundry_text_edit_class_init (FoundryTextEditClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_text_edit_finalize;
}

static void
foundry_text_edit_init (FoundryTextEdit *self)
{
}

FoundryTextEdit *
foundry_text_edit_new (GFile      *file,
                       guint       begin_line,
                       int         begin_line_offset,
                       guint       end_line,
                       int         end_line_offset,
                       const char *replacement)
{
  FoundryTextEdit *self;

  g_return_val_if_fail (!file || G_IS_FILE (file), NULL);

  self = g_object_new (FOUNDRY_TYPE_TEXT_EDIT, NULL);
  self->begin_line = begin_line;
  self->end_line = end_line;
  self->begin_line_offset = MAX (-1, begin_line_offset);
  self->end_line_offset = MAX (-1, end_line_offset);
  self->replacement = g_strdup (replacement);

  g_set_object (&self->file, file);

  return self;
}

/**
 * foundry_text_edit_dup_file:
 * @self: a #FoundryTextEdit
 *
 * Gets the underlying #GFile if any.
 *
 * Returns: (transfer full) (nullable): a #GFile or %NULL
 */
GFile *
foundry_text_edit_dup_file (FoundryTextEdit *self)
{
  GFile *ret = NULL;

  g_return_val_if_fail (FOUNDRY_IS_TEXT_EDIT (self), NULL);

  g_set_object (&ret, self->file);

  return ret;
}

/**
 * foundry_text_edit_dup_replacement:
 * @self: a #FoundryTextEdit
 *
 * Gets the replacement text if any.
 *
 * Returns: (transfer full) (nullable): a string or %NULL
 */
char *
foundry_text_edit_dup_replacement (FoundryTextEdit *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEXT_EDIT (self), NULL);

  return g_strdup (self->replacement);
}

void
foundry_text_edit_get_range (FoundryTextEdit *self,
                             guint           *begin_line,
                             int             *begin_line_offset,
                             guint           *end_line,
                             int             *end_line_offset)
{
  g_return_if_fail (FOUNDRY_IS_TEXT_EDIT (self));

  if (begin_line != NULL)
    *begin_line = self->begin_line;

  if (begin_line_offset != NULL)
    *begin_line_offset = self->begin_line_offset;

  if (end_line != NULL)
    *end_line = self->end_line;

  if (end_line_offset != NULL)
    *end_line_offset = self->end_line_offset;
}

int
foundry_text_edit_compare (const FoundryTextEdit *a,
                           const FoundryTextEdit *b)
{
  if (!g_file_equal (a->file, b->file))
    {
      g_autofree char *uri_a = g_file_get_uri (a->file);
      g_autofree char *uri_b = g_file_get_uri (b->file);

      return g_strcmp0 (uri_a, uri_b);
    }

  if (a->begin_line < b->begin_line)
    return -1;
  else if (a->begin_line > b->begin_line)
    return 1;

  if (a->begin_line_offset < b->begin_line_offset)
    return -1;
  else if (a->begin_line_offset > b->begin_line_offset)
    return 1;

  /* Longer runs first */
  if (a->end_line < b->end_line)
    return 1;
  else if (a->end_line > b->end_line)
    return -1;

  if (a->end_line_offset < b->end_line_offset)
    return 1;
  else if (a->end_line_offset > b->end_line_offset)
    return -1;

  return 0;
}
