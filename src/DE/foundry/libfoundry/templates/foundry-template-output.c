/* foundry-template-output.c
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "foundry-template-output.h"

struct _FoundryTemplateOutput
{
  GObject parent_instance;
  GFile *file;
  GBytes *contents;
  int mode;
  guint is_dir : 1;
};

G_DEFINE_FINAL_TYPE (FoundryTemplateOutput, foundry_template_output, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_CONTENTS,
  PROP_FILE,
  PROP_MODE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_template_output_finalize (GObject *object)
{
  FoundryTemplateOutput *self = (FoundryTemplateOutput *)object;

  g_clear_object (&self->file);
  g_clear_pointer (&self->contents, g_bytes_unref);

  G_OBJECT_CLASS (foundry_template_output_parent_class)->finalize (object);
}

static void
foundry_template_output_get_property (GObject    *object,
                                      guint       prop_id,
                                      GValue     *value,
                                      GParamSpec *pspec)
{
  FoundryTemplateOutput *self = FOUNDRY_TEMPLATE_OUTPUT (object);

  switch (prop_id)
    {
    case PROP_CONTENTS:
      g_value_take_boxed (value, foundry_template_output_dup_contents (self));
      break;

    case PROP_FILE:
      g_value_take_object (value, foundry_template_output_dup_file (self));
      break;

    case PROP_MODE:
      g_value_set_int (value, foundry_template_output_get_mode (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_template_output_class_init (FoundryTemplateOutputClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_template_output_finalize;
  object_class->get_property = foundry_template_output_get_property;

  properties[PROP_CONTENTS] =
    g_param_spec_boxed ("contents", NULL, NULL,
                        G_TYPE_BYTES,
                        (G_PARAM_READABLE |
                         G_PARAM_STATIC_STRINGS));

  properties[PROP_FILE] =
    g_param_spec_object ("file", NULL, NULL,
                         G_TYPE_FILE,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_MODE] =
    g_param_spec_int ("mode", NULL, NULL,
                      -1, G_MAXINT, 0,
                      (G_PARAM_READABLE |
                       G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_template_output_init (FoundryTemplateOutput *self)
{
  self->mode = -1;
}

/**
 * foundry_template_output_new:
 * @file: a [iface@Gio.File] to write to
 * @contents: the contents to write as an immutable byte buffer
 * @mode: the file mode (or -1 to use the default)
 *
 * Returns: (transfer full):
 */
FoundryTemplateOutput *
foundry_template_output_new (GFile  *file,
                             GBytes *contents,
                             int     mode)
{
  static const char empty[1] = {0};
  FoundryTemplateOutput *self;

  g_return_val_if_fail (G_IS_FILE (file), NULL);
  g_return_val_if_fail (mode >= -1, NULL);

  self = g_object_new (FOUNDRY_TYPE_TEMPLATE_OUTPUT, NULL);
  self->file = g_object_ref (file);
  self->contents = contents ? g_bytes_ref (contents) : g_bytes_new_static (empty, 0);
  self->mode = mode;

  return self;
}

FoundryTemplateOutput *
foundry_template_output_new_directory (GFile *file)
{
  FoundryTemplateOutput *self;

  g_return_val_if_fail (G_IS_FILE (file), NULL);

  self = g_object_new (FOUNDRY_TYPE_TEMPLATE_OUTPUT, NULL);
  self->file = g_object_ref (file);
  self->is_dir = TRUE;

  return self;
}

/**
 * foundry_template_output_dup_contents:
 * @self: a [class@Foundry.TemplateOutput]
 *
 * Returns: (transfer full):
 */
GBytes *
foundry_template_output_dup_contents (FoundryTemplateOutput *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEMPLATE_OUTPUT (self), NULL);

  return g_bytes_ref (self->contents);
}

/**
 * foundry_template_output_dup_file:
 * @self: a [class@Foundry.TemplateOutput]
 *
 * Returns: (transfer full):
 */
GFile *
foundry_template_output_dup_file (FoundryTemplateOutput *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEMPLATE_OUTPUT (self), NULL);

  return g_object_ref (self->file);
}

/**
 * foundry_template_output_get_mode:
 * @self: a [class@Foundry.TemplateOutput]
 *
 * The mode for the file or -1 to ignore the mode.
 *
 * Returns:
 */
int
foundry_template_output_get_mode (FoundryTemplateOutput *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TEMPLATE_OUTPUT (self), 0);

  return self->mode;
}

static DexFuture *
foundry_template_output_write_contents (DexFuture *completed,
                                        gpointer   user_data)
{
  FoundryTemplateOutput *self = FOUNDRY_TEMPLATE_OUTPUT (user_data);

  return dex_file_replace_contents_bytes (self->file,
                                          self->contents,
                                          NULL,
                                          FALSE,
                                          G_FILE_CREATE_REPLACE_DESTINATION);
}

static DexFuture *
foundry_template_output_set_mode (DexFuture *completed,
                                  gpointer   user_data)
{
  FoundryTemplateOutput *self = FOUNDRY_TEMPLATE_OUTPUT (user_data);
  g_autoptr(GFileInfo) info = g_file_info_new ();

  g_file_info_set_attribute_uint32 (info, G_FILE_ATTRIBUTE_UNIX_MODE, self->mode);

  return dex_file_set_attributes (self->file, info, 0, 0);
}

/**
 * foundry_template_output_write:
 * @self: a [class@Foundry.TemplateOutput]
 *
 * Writes the output contents to the destination file.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value or rejects with error.
 */
DexFuture *
foundry_template_output_write (FoundryTemplateOutput *self)
{
  g_autoptr(GFile) directory = NULL;
  DexFuture *future;

  dex_return_error_if_fail (FOUNDRY_IS_TEMPLATE_OUTPUT (self));

  if (self->is_dir)
    return dex_file_make_directory_with_parents (self->file);

  directory = g_file_get_parent (self->file);

  future = dex_file_make_directory_with_parents (directory);
  future = dex_future_then (future,
                            foundry_template_output_write_contents,
                            g_object_ref (self),
                            g_object_unref);

  if (self->mode > -1)
    future = dex_future_then (future,
                              foundry_template_output_set_mode,
                              g_object_ref (self),
                              g_object_unref);

  return future;
}
