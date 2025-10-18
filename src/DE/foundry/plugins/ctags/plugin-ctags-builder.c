/* plugin-ctags-builder.c
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

#include <errno.h>

#include <glib/gstdio.h>

#include "plugin-ctags-builder.h"

struct _PluginCtagsBuilder
{
  GObject    parent_instance;
  GFile     *destination;
  GFile     *options_file;
  GPtrArray *files;
  char      *ctags;
};

G_DEFINE_FINAL_TYPE (PluginCtagsBuilder, plugin_ctags_builder, G_TYPE_OBJECT)

static void
plugin_ctags_builder_finalize (GObject *object)
{
  PluginCtagsBuilder *self = (PluginCtagsBuilder *)object;

  g_clear_object (&self->destination);
  g_clear_object (&self->options_file);
  g_clear_pointer (&self->files, g_ptr_array_unref);
  g_clear_pointer (&self->ctags, g_free);

  G_OBJECT_CLASS (plugin_ctags_builder_parent_class)->finalize (object);
}

static void
plugin_ctags_builder_class_init (PluginCtagsBuilderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = plugin_ctags_builder_finalize;
}

static void
plugin_ctags_builder_init (PluginCtagsBuilder *self)
{
  g_set_str (&self->ctags, "ctags");
}

PluginCtagsBuilder *
plugin_ctags_builder_new (GFile *destination)
{
  PluginCtagsBuilder *self;

  g_return_val_if_fail (G_IS_FILE (destination), NULL);

  self = g_object_new (PLUGIN_TYPE_CTAGS_BUILDER, NULL);
  self->destination = g_object_ref (destination);

  return self;
}

void
plugin_ctags_builder_add_file (PluginCtagsBuilder *self,
                               GFile              *file)
{
  g_return_if_fail (PLUGIN_IS_CTAGS_BUILDER (self));
  g_return_if_fail (G_IS_FILE (file));
  g_return_if_fail (g_file_is_native (file));

  if (self->files == NULL)
    self->files = g_ptr_array_new_with_free_func (g_object_unref);

  g_ptr_array_add (self->files, g_object_ref (file));
}

static void
remove_tmpfile (GFile *tmp_file)
{
  dex_future_disown (dex_file_delete (tmp_file, 0));
}

static DexFuture *
plugin_ctags_builder_build_fiber (gpointer data)
{
  PluginCtagsBuilder *self = data;
  g_autoptr(GSubprocessLauncher) launcher = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GPtrArray) argv = NULL;
  g_autoptr(GString) str = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GBytes) bytes = NULL;
  g_autoptr(GFile) dir = NULL;
  g_autoptr(GFile) tmp_file = NULL;
  g_autofree char *ctags = NULL;
  g_autofd int tmp_fd = -1;
  GOutputStream *stdin_stream = NULL;
  g_autofree char *tmpl = NULL;
  const char *cwd;

  dex_return_error_if_fail (PLUGIN_IS_CTAGS_BUILDER (self));

  if (self->files == NULL || self->files->len == 0)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_FAILED,
                                  "No files to index");

  if (!g_file_is_native (self->destination))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_FAILED,
                                  "Destination is not a native file");

  ctags = g_strdup (self->ctags);
  dir = g_file_get_parent (self->destination);
  tmpl = g_build_filename (g_file_peek_path (dir), "tags.XXXXXX", NULL);
  cwd = g_file_peek_path (dir);
  argv = g_ptr_array_new ();

  if (!dex_await_boolean (dex_file_query_exists (dir), NULL))
    {
      if (!dex_await (dex_file_make_directory_with_parents (dir), &error))
        return dex_future_new_for_error (g_steal_pointer (&error));
    }

  if ((tmp_fd = g_mkstemp (tmpl)) == -1)
    return dex_future_new_for_errno (errno);

  tmp_file = g_file_new_for_path (tmpl);

  launcher = g_subprocess_launcher_new (G_SUBPROCESS_FLAGS_STDIN_PIPE | G_SUBPROCESS_FLAGS_STDERR_SILENCE);

  g_subprocess_launcher_set_cwd (launcher, cwd);
  g_subprocess_launcher_setenv (launcher, "TMPDIR", cwd, TRUE);
  g_subprocess_launcher_take_fd (launcher, g_steal_fd (&tmp_fd), STDOUT_FILENO);

#ifdef __linux__
  g_ptr_array_add (argv, (char *)"nice");
#endif

  g_ptr_array_add (argv, (char *)ctags);
  g_ptr_array_add (argv, (char *)"-f");
  g_ptr_array_add (argv, (char *)"-");
  g_ptr_array_add (argv, (char *)"--tag-relative=no");
  g_ptr_array_add (argv, (char *)"--exclude=.git");
  g_ptr_array_add (argv, (char *)"--exclude=.bzr");
  g_ptr_array_add (argv, (char *)"--exclude=.svn");
  g_ptr_array_add (argv, (char *)"--exclude=.flatpak-builder");
  g_ptr_array_add (argv, (char *)"--sort=yes");
  g_ptr_array_add (argv, (char *)"--languages=all");
  g_ptr_array_add (argv, (char *)"--extra=+F");
  g_ptr_array_add (argv, (char *)"--c-kinds=+defgpstx");

  if (self->options_file && g_file_is_native (self->options_file))
    {
      g_ptr_array_add (argv, (char *)"--options");
      g_ptr_array_add (argv, (char *)g_file_peek_path (self->options_file));
    }

  g_ptr_array_add (argv, (char *)"-L");
  g_ptr_array_add (argv, (char *)"-");
  g_ptr_array_add (argv, NULL);

  if (!(subprocess = g_subprocess_launcher_spawnv (launcher, (const char * const *)argv->pdata, &error)))
    {
      remove_tmpfile (tmp_file);
      return dex_future_new_for_error (g_steal_pointer (&error));
    }

  stdin_stream = g_subprocess_get_stdin_pipe (subprocess);
  str = g_string_new (NULL);

  for (guint i = 0; i < self->files->len; i++)
    {
      GFile *file = g_ptr_array_index (self->files, i);

      g_string_append_printf (str, "%s\n", g_file_peek_path (file));
    }

  bytes = g_string_free_to_bytes (g_steal_pointer (&str));

  if (!dex_await (dex_output_stream_write_bytes (stdin_stream, bytes, G_PRIORITY_DEFAULT), &error))
    {
      remove_tmpfile (tmp_file);
      return dex_future_new_for_error (g_steal_pointer (&error));
    }

  dex_future_disown (dex_output_stream_close (stdin_stream, 0));

  if (!dex_await (dex_subprocess_wait_check (subprocess), &error))
    {
      remove_tmpfile (tmp_file);
      return dex_future_new_for_error (g_steal_pointer (&error));
    }

  return dex_file_move (tmp_file, self->destination, G_FILE_COPY_OVERWRITE, 0, NULL, NULL, NULL);
}

DexFuture *
plugin_ctags_builder_build (PluginCtagsBuilder *self)
{
  return dex_scheduler_spawn (NULL, 0,
                              plugin_ctags_builder_build_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

void
plugin_ctags_builder_set_options_file (PluginCtagsBuilder *self,
                                       GFile              *options_file)
{
  g_return_if_fail (PLUGIN_IS_CTAGS_BUILDER (self));
  g_return_if_fail (!options_file || G_IS_FILE (options_file));

  g_set_object (&self->options_file, options_file);
}

void
plugin_ctags_builder_set_ctags_path (PluginCtagsBuilder *self,
                                     const char         *ctags_path)
{
  g_return_if_fail (PLUGIN_IS_CTAGS_BUILDER (self));

  if (ctags_path == NULL || ctags_path[0] == 0)
    ctags_path = "ctags";

  g_set_str (&self->ctags, ctags_path);
}
