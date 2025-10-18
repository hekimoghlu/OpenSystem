/* foundry-file.c
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

#include <errno.h>

#include <glib/gstdio.h>

#include "foundry-file.h"
#include "foundry-process-launcher.h"
#include "foundry-util-private.h"

static DexFuture *
foundry_file_find_in_ancestors_fiber (GFile      *file,
                                      const char *name)
{
  GFile *parent;

  g_assert (G_IS_FILE (file));
  g_assert (name != NULL);

  parent = g_file_get_parent (file);

  while (parent != NULL)
    {
      g_autoptr(GFile) child = g_file_get_child (parent, name);
      g_autoptr(GFile) old_parent = NULL;

      if (dex_await_boolean (dex_file_query_exists (child), NULL))
        return dex_future_new_true ();

      old_parent = g_steal_pointer (&parent);
      parent = g_file_get_parent (old_parent);
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Failed to locate \"%s\" within ancestors",
                                name);
}

/**
 * foundry_file_find_in_ancestors:
 * @file: a [iface@Gio.File]
 * @name: the name of the file to find in the ancestors such as ".gitignore"
 *
 * Locates @name within any of the ancestors of @file up to the root of
 * the filesystem.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [iface@Gio.File] or rejects with error,
 */
DexFuture *
foundry_file_find_in_ancestors (GFile      *file,
                                const char *name)
{
  dex_return_error_if_fail (G_IS_FILE (file));
  dex_return_error_if_fail (name != NULL);

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_file_find_in_ancestors_fiber),
                                  2,
                                  G_TYPE_FILE, file,
                                  G_TYPE_STRING, name);
}

static gboolean
is_internally_ignored (const char *name)
{
  if (name == NULL)
    return TRUE;

  if (g_str_has_prefix (name, ".goutputstream-"))
    return TRUE;

  if (g_str_has_suffix (name, "~"))
    return TRUE;

  if (g_str_has_suffix (name, ".min.js") || strstr (name, ".min.js.") != NULL)
    return TRUE;

  return FALSE;
}

typedef struct _Find
{
  GFile        *file;
  GPatternSpec *spec;
  GRegex       *regex;
  guint         depth;
} Find;

static void
populate_descendants_matching (GFile        *file,
                               GCancellable *cancellable,
                               GPtrArray    *results,
                               const Find   *find,
                               guint         depth)
{
  g_autoptr(GFileEnumerator) enumerator = NULL;
  g_autoptr(GPtrArray) children = NULL;

  g_assert (G_IS_FILE (file));
  g_assert (results != NULL);
  g_assert (find != NULL);
  g_assert (find->regex || find->spec);
  g_assert (!cancellable || G_IS_CANCELLABLE (cancellable));

  if (depth == 0)
    return;

  enumerator = g_file_enumerate_children (file,
                                          G_FILE_ATTRIBUTE_STANDARD_NAME","
                                          G_FILE_ATTRIBUTE_STANDARD_IS_SYMLINK","
                                          G_FILE_ATTRIBUTE_STANDARD_TYPE,
                                          G_FILE_QUERY_INFO_NONE,
                                          cancellable,
                                          NULL);

  if (enumerator == NULL)
    return;

  for (;;)
    {
      g_autoptr(GFileInfo) info = g_file_enumerator_next_file (enumerator, cancellable, NULL);
      const gchar *name;
      GFileType file_type;

      if (info == NULL)
        break;

      name = g_file_info_get_name (info);
      file_type = g_file_info_get_file_type (info);

      if (is_internally_ignored (name))
        continue;

      if ((find->spec && g_pattern_spec_match_string (find->spec, name)) ||
          (find->regex && g_regex_match (find->regex, name, 0, NULL)))
        g_ptr_array_add (results, g_file_enumerator_get_child (enumerator, info));

      if (!g_file_info_get_is_symlink (info) && file_type == G_FILE_TYPE_DIRECTORY)
        {
          /* Try to project ourselves a bit from common traps */
          if (g_strcmp0 (name, ".flatpak-builder") == 0 ||
              g_strcmp0 (name, ".cache") == 0)
            continue;

          if (children == NULL)
            children = g_ptr_array_new_with_free_func (g_object_unref);

          g_ptr_array_add (children, g_file_enumerator_get_child (enumerator, info));
        }
    }

  g_file_enumerator_close (enumerator, cancellable, NULL);

  if (children != NULL)
    {
      for (guint i = 0; i < children->len; i++)
        {
          GFile *child = g_ptr_array_index (children, i);

          populate_descendants_matching (child, cancellable, results, find, depth - 1);
        }
    }
}

static DexFuture *
find_matching_fiber (gpointer user_data)
{
  Find *find = user_data;
  g_autoptr(GPtrArray) ar = NULL;

  g_assert (find != NULL);
  g_assert (G_IS_FILE (find->file));
  g_assert (find->spec || find->regex);
  g_assert (find->depth > 0);

  ar = g_ptr_array_new_with_free_func (g_object_unref);

  populate_descendants_matching (find->file, NULL, ar, find, find->depth);

  return dex_future_new_take_boxed (G_TYPE_PTR_ARRAY, g_steal_pointer (&ar));
}

static void
find_free (Find *find)
{
  g_clear_object (&find->file);
  g_clear_pointer (&find->spec, g_pattern_spec_free);
  g_clear_pointer (&find->regex, g_regex_unref);
  g_free (find);
}

static DexFuture *
find_matching (GFile        *file,
               GPatternSpec *spec,
               guint         depth)
{
  Find *state;

  g_assert (G_IS_FILE (file));
  g_assert (spec != NULL);
  g_assert (depth > 0);

  state = g_new0 (Find, 1);
  state->file = g_object_ref (file);
  state->spec = g_pattern_spec_copy (spec);
  state->depth = depth;

  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              find_matching_fiber,
                              state,
                              (GDestroyNotify) find_free);

}

/**
 * foundry_file_find_with_depth:
 * @file: an [iface@Gio.File]
 * @pattern: the pattern to find
 * @max_depth: the max depth to recurse
 *
 * Locates files starting from @file matching @pattern.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves
 *   to a [struct@GLib.PtrArray] of [iface@Gio.File].
 */
DexFuture *
foundry_file_find_with_depth (GFile       *file,
                              const gchar *pattern,
                              guint        max_depth)
{
  g_autoptr(GPatternSpec) spec = NULL;

  dex_return_error_if_fail (G_IS_FILE (file));
  dex_return_error_if_fail (pattern != NULL);

  if (!(spec = g_pattern_spec_new (pattern)))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_INVAL,
                                  "Invalid pattern");

  if (max_depth == 0)
    max_depth = G_MAXUINT;

  return find_matching (file, spec, max_depth);
}

/**
 * foundry_file_find_regex_with_depth:
 * @file: an [iface@Gio.File]
 * @regex: A regex to match filenames within descendants
 * @max_depth: the max depth to recurse
 *
 * Locates files starting from @file matching @regex.
 *
 * The regex will be passed the name within the parent directory, not the enter
 * path from @file.
 *
 * Returns: (transfer full): a [class@Dex.Future]
 */
DexFuture *
foundry_file_find_regex_with_depth (GFile  *file,
                                    GRegex *regex,
                                    guint   max_depth)
{
  g_autoptr(GPatternSpec) spec = NULL;
  Find *state;

  dex_return_error_if_fail (G_IS_FILE (file));
  dex_return_error_if_fail (regex != NULL);

  if (max_depth == 0)
    max_depth = G_MAXUINT;

  state = g_new0 (Find, 1);
  state->file = g_object_ref (file);
  state->spec = NULL;
  state->regex = g_regex_ref (regex);
  state->depth = max_depth;

  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              find_matching_fiber,
                              state,
                              (GDestroyNotify) find_free);
}

/**
 * foundry_file_query_exists_nofollow:
 * @file: a [iface@Gio.File]
 *
 * Resolves to true if @file exists.
 *
 * Does not follow symlink.
 *
 * Returns: (transfer full): a future that resolves to a boolean
 */
DexFuture *
foundry_file_query_exists_nofollow (GFile *file)
{
  dex_return_error_if_fail (G_IS_FILE (file));

  return dex_future_then (dex_file_query_info (file,
                                               G_FILE_ATTRIBUTE_STANDARD_TYPE,
                                               G_FILE_QUERY_INFO_NOFOLLOW_SYMLINKS,
                                               G_PRIORITY_DEFAULT),
                          foundry_future_return_true,
                          NULL, NULL);
}

/* NOTE: This requires that file exists */
/**
 * foundry_file_canonicalize:
 * @file: a [iface@Gio.File]
 *
 * Returns: (transfer full):
 */
GFile *
foundry_file_canonicalize (GFile   *file,
                           GError **error)
{
  char *canonical_path = NULL;

  if ((canonical_path = realpath (g_file_peek_path (file), NULL)))
    {
      GFile *ret = g_file_new_for_path (canonical_path);
      free (canonical_path);
      return ret;
    }

  return (gpointer)foundry_set_error_from_errno (error);
}

/* NOTE: This requires both files to exist */
gboolean
foundry_file_is_in (GFile *file,
                    GFile *toplevel)
{
  g_autoptr(GFile) canonical_file = NULL;
  g_autoptr(GFile) canonical_toplevel = NULL;

  if (!(canonical_toplevel = foundry_file_canonicalize (toplevel, NULL)))
    return FALSE;

  if (!(canonical_file = foundry_file_canonicalize (file, NULL)))
    return FALSE;

  return g_file_equal (canonical_file, canonical_toplevel) ||
         g_file_has_prefix (canonical_file, canonical_toplevel);
}

typedef struct _ListChildrenTyped
{
  GFile     *file;
  char      *attributes;
  GFileType  file_type;
} ListChildrenTyped;

static void
list_children_typed_free (gpointer data)
{
  ListChildrenTyped *state = data;

  g_clear_object (&state->file);
  g_clear_pointer (&state->attributes, g_free);
  g_free (state);
}

static DexFuture *
foundry_list_children_typed_fiber (gpointer user_data)
{
  ListChildrenTyped *state = user_data;
  g_autoptr(GFileEnumerator) enumerator = NULL;
  g_autoptr(GPtrArray) ar = NULL;
  g_autoptr(GError) error = NULL;
  GList *list;

  g_assert (state != NULL);
  g_assert (G_IS_FILE (state->file));

  if (!(enumerator = dex_await_object (dex_file_enumerate_children (state->file,
                                                                    state->attributes,
                                                                    G_FILE_QUERY_INFO_NONE,
                                                                    G_PRIORITY_DEFAULT),
                                       &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  ar = g_ptr_array_new_with_free_func (g_object_unref);

  while ((list = dex_await_boxed (dex_file_enumerator_next_files (enumerator,
                                                                  100,
                                                                  G_PRIORITY_DEFAULT),
                                  &error)))
    {
      for (const GList *iter = list; iter; iter = iter->next)
        {
          GFileInfo *info = iter->data;
          GFileType file_type = g_file_info_get_file_type (info);

          if (file_type == state->file_type)
            g_ptr_array_add (ar, iter->data);
          else
            g_object_unref (iter->data);
        }

      g_list_free (list);
    }

  if (error != NULL)
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_take_boxed (G_TYPE_PTR_ARRAY, g_steal_pointer (&ar));
}

/**
 * foundry_file_list_children_typed:
 * @file: a [iface@Gio.File]
 * @file_type:
 * @attributes:
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [struct@GLib.PtrArray] of [iface@Gio.File] or rejects with error.
 */
DexFuture *
foundry_file_list_children_typed (GFile      *file,
                                  GFileType   file_type,
                                  const char *attributes)
{
  ListChildrenTyped *state;

  g_return_val_if_fail (G_IS_FILE (file), NULL);

  state = g_new0 (ListChildrenTyped, 1);
  state->file = g_object_ref (file);
  state->file_type = file_type;

  if (attributes == NULL)
    state->attributes = g_strdup (G_FILE_ATTRIBUTE_STANDARD_NAME","G_FILE_ATTRIBUTE_STANDARD_TYPE);
  else
    state->attributes = g_strdup_printf ("%s,%s,%s",
                                         G_FILE_ATTRIBUTE_STANDARD_NAME,
                                         G_FILE_ATTRIBUTE_STANDARD_TYPE,
                                         attributes);

  return dex_scheduler_spawn (NULL, 0,
                              foundry_list_children_typed_fiber,
                              state,
                              list_children_typed_free);
}

static DexFuture *
foundry_host_file_get_contents_bytes_fiber (gpointer data)
{
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GBytes) bytes = NULL;
  g_autoptr(GFile) tmpgfile = NULL;
  g_autofree char *tmpfile = NULL;
  const char *path = data;
  g_autofd int fd = -1;

  g_assert (path != NULL);
  g_assert (g_path_is_absolute (path));

  tmpfile = g_build_filename (g_get_tmp_dir (), ".foundry-host-file-XXXXXX", NULL);
  tmpgfile = g_file_new_for_path (tmpfile);

  /* We open a FD locally that we can write to and then pass that as our
   * stdout across the boundary so we can avoid incrementally reading
   * and instead do it once at the end.
   */
  if (-1 == (fd = g_mkstemp (tmpfile)))
    {
      int errsv = errno;
      return dex_future_new_reject (G_FILE_ERROR,
                                    g_file_error_from_errno (errsv),
                                    "%s", g_strerror (errsv));
    }

  launcher = foundry_process_launcher_new ();
  foundry_process_launcher_push_host (launcher);
  foundry_process_launcher_take_fd (launcher, g_steal_fd (&fd), STDOUT_FILENO);
  foundry_process_launcher_append_argv (launcher, "cat");
  foundry_process_launcher_append_argv (launcher, path);

  if ((subprocess = foundry_process_launcher_spawn_with_flags (launcher,
                                                               G_SUBPROCESS_FLAGS_STDERR_SILENCE,
                                                               &error)) &&
      dex_await (dex_subprocess_wait_check (subprocess), &error))
    bytes = dex_await_boxed (dex_file_load_contents_bytes (tmpgfile), &error);

  dex_await (dex_file_delete (tmpgfile, G_PRIORITY_DEFAULT), NULL);

  g_assert (bytes != NULL || error != NULL);

  if (error != NULL)
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_take_boxed (G_TYPE_BYTES, g_steal_pointer (&bytes));
}

/**
 * foundry_host_file_get_contents_bytes:
 * @path: a path to a file on the host
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [struct@GLib.Bytes] or rejects with error.
 */
DexFuture *
foundry_host_file_get_contents_bytes (const char *path)
{
  dex_return_error_if_fail (path != NULL);
  dex_return_error_if_fail (g_path_is_absolute (path));

  if (!_foundry_in_container ())
    {
      g_autoptr(GFile) file = g_file_new_for_path (path);

      return dex_file_load_contents_bytes (file);
    }

  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              foundry_host_file_get_contents_bytes_fiber,
                              g_strdup (path),
                              g_free);
}
