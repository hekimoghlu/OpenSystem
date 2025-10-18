/* foundry-util.c
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

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif

#include <errno.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <glib/gi18n-lib.h>
#include <glib/gstdio.h>
#include <gio/gio.h>

#include "line-reader-private.h"

#include "foundry-model-manager.h"
#include "foundry-path.h"
#include "foundry-triplet.h"
#include "foundry-util-private.h"

static char **
get_environ_from_stdout (GSubprocess *subprocess)
{
  g_autofree char *stdout_buf = NULL;

  if (g_subprocess_communicate_utf8 (subprocess, NULL, NULL, &stdout_buf, NULL, NULL))
    {
      g_autoptr(GPtrArray) env = g_ptr_array_new_with_free_func (g_free);
      LineReader reader;
      gsize line_len;
      char *line;

      line_reader_init (&reader, stdout_buf, -1);

      while ((line = line_reader_next (&reader, &line_len)))
        {
          line[line_len] = 0;

          if (!g_ascii_isalpha (*line) && *line != '_')
            continue;

          for (const char *iter = line; *iter; iter = g_utf8_next_char (iter))
            {
              if (*iter == '=')
                {
                  g_ptr_array_add (env, g_strdup (line));
                  break;
                }

              if (!g_ascii_isalnum (*iter) && *iter != '_')
                break;
            }
        }

      if (env->len > 0)
        {
          g_ptr_array_add (env, NULL);
          return (char **)g_ptr_array_free (g_steal_pointer (&env), FALSE);
        }
    }

  return NULL;
}

gboolean
_foundry_in_container (void)
{
  static gsize initialized;
  static gboolean in_container;

  if (g_once_init_enter (&initialized))
    {
      in_container = g_file_test ("/.flatpak-info", G_FILE_TEST_EXISTS) ||
                     g_file_test ("/var/run/.containerenv", G_FILE_TEST_EXISTS);
      g_once_init_leave (&initialized, TRUE);
    }

  return in_container;
}

const char * const *
_foundry_host_environ (void)
{
  static char **host_environ;

  if (host_environ == NULL)
    {
      if (_foundry_in_container ())
        {
          g_autoptr(GSubprocessLauncher) launcher = NULL;
          g_autoptr(GSubprocess) subprocess = NULL;
          g_autoptr(GError) error = NULL;

          launcher = g_subprocess_launcher_new (G_SUBPROCESS_FLAGS_STDOUT_PIPE);
          subprocess = g_subprocess_launcher_spawn (launcher, &error,
                                                    "flatpak-spawn", "--host", "printenv", NULL);
          if (subprocess != NULL)
            host_environ = get_environ_from_stdout (subprocess);
        }

      if (host_environ == NULL)
        host_environ = g_get_environ ();
    }

  return (const char * const *)host_environ;
}

char *
_foundry_create_host_triplet (const char *arch,
                              const char *kernel,
                              const char *system)
{
  if (arch == NULL || kernel == NULL)
    return g_strdup (_foundry_get_system_type ());
  else if (system == NULL)
    return g_strdup_printf ("%s-%s", arch, kernel);
  else
    return g_strdup_printf ("%s-%s-%s", arch, kernel, system);
}

const char *
_foundry_get_system_type (void)
{
  static char *system_type;
  g_autofree char *os_lower = NULL;
  const char *machine = NULL;
  struct utsname u;

  if (system_type != NULL)
    return system_type;

  if (uname (&u) < 0)
    return g_strdup ("unknown");

  os_lower = g_utf8_strdown (u.sysname, -1);

  /* config.sub doesn't accept amd64-OS */
  machine = strcmp (u.machine, "amd64") ? u.machine : "x86_64";

  /*
   * TODO: Clearly we want to discover "gnu", but that should be just fine
   *       for a default until we try to actually run on something non-gnu.
   *       Which seems unlikely at the moment. If you run FreeBSD, you can
   *       probably fix this for me :-) And while you're at it, make the
   *       uname() call more portable.
   */

#ifdef __GLIBC__
  system_type = g_strdup_printf ("%s-%s-%s", machine, os_lower, "gnu");
#else
  system_type = g_strdup_printf ("%s-%s", machine, os_lower);
#endif

  return system_type;
}

char *
_foundry_get_system_arch (void)
{
  static GHashTable *remap;
  const char *machine;
  struct utsname u;

  if (uname (&u) < 0)
    return g_strdup ("unknown");

  if (g_once_init_enter (&remap))
    {
      GHashTable *mapping;

      mapping = g_hash_table_new (g_str_hash, g_str_equal);
      g_hash_table_insert (mapping, (char *)"amd64", (char *)"x86_64");
      g_hash_table_insert (mapping, (char *)"armv7l", (char *)"aarch64");
      g_hash_table_insert (mapping, (char *)"i686", (char *)"i386");

      g_once_init_leave (&remap, mapping);
    }

  if (g_hash_table_lookup_extended (remap, u.machine, NULL, (gpointer *)&machine))
    return g_strdup (machine);
  else
    return g_strdup (u.machine);
}

void
_foundry_fd_write_all (int         fd,
                       const char *message,
                       gssize      to_write)
{
  const char *data = message;

  if (fd < 0)
    return;

  if (to_write < 0)
    to_write = strlen (message);

  while (to_write > 0)
    {
      gssize n_written;

      errno = 0;
      n_written = write (fd, data, to_write);

      if (n_written < 0)
        return;

      if (n_written == 0 && errno == EINTR)
        continue;

      to_write -= n_written;
      data += (gsize)n_written;
    }
}

typedef struct
{
  char *tmpdir;
  char *template_name;
  int mode;
} Mkdtemp;

static void
mkdtemp_free (Mkdtemp *state)
{
  g_clear_pointer (&state->tmpdir, g_free);
  g_clear_pointer (&state->template_name, g_free);
  g_free (state);
}

static DexFuture *
_foundry_mkdtemp_fiber (gpointer data)
{
  static const char letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  static const guint n_letters = sizeof letters - 1;

  Mkdtemp *state = data;
  g_autofree char *copy = g_strdup (state->template_name);
  char *XXXXXX = strstr (copy, "XXXXXX");

  if (XXXXXX == NULL)
    return dex_future_new_reject (G_FILE_ERROR,
                                  G_FILE_ERROR_INVAL,
                                  "Invalid template name %s",
                                  state->template_name);

  for (guint count = 0; count < 100; count++)
    {
      gint64 v = g_get_real_time () ^ g_random_int ();
      g_autofree char *path = NULL;

      XXXXXX[0] = letters[v % n_letters];
      v /= n_letters;
      XXXXXX[1] = letters[v % n_letters];
      v /= n_letters;
      XXXXXX[2] = letters[v % n_letters];
      v /= n_letters;
      XXXXXX[3] = letters[v % n_letters];
      v /= n_letters;
      XXXXXX[4] = letters[v % n_letters];
      v /= n_letters;
      XXXXXX[5] = letters[v % n_letters];

      path = g_build_filename (state->tmpdir, copy, NULL);

      errno = 0;

      if (g_mkdir (path, state->mode) == 0)
        return dex_future_new_for_string (g_steal_pointer (&path));

      if (errno != EEXIST)
        {
          int errsv = errno;
          return dex_future_new_reject (G_FILE_ERROR,
                                        g_file_error_from_errno (errsv),
                                        "%s",
                                        g_strerror (errsv));
        }
    }

  return dex_future_new_reject (G_FILE_ERROR,
                                G_FILE_ERROR_EXIST,
                                "%s",
                                g_strerror (EEXIST));
}

DexFuture *
_foundry_mkdtemp (const char *tmpdir,
                  const char *template_name)
{
  Mkdtemp *state;

  g_return_val_if_fail (tmpdir != NULL, NULL);
  g_return_val_if_fail (template_name != NULL, NULL);
  g_return_val_if_fail (strstr (template_name, "XXXXXX") != NULL, NULL);

  state = g_new0 (Mkdtemp, 1);
  state->tmpdir = g_strdup (tmpdir);
  state->template_name = g_strdup (template_name);
  state->mode = 0770;

  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              _foundry_mkdtemp_fiber,
                              state,
                              (GDestroyNotify)mkdtemp_free);
}

const char *
foundry_get_default_arch (void)
{
  static const char *default_arch;

  if (g_once_init_enter (&default_arch))
    {
      g_autoptr(FoundryTriplet) triplet = foundry_triplet_new_from_system ();
      const char *value = g_intern_string (foundry_triplet_get_arch (triplet));
      g_once_init_leave (&default_arch, value);
    }

  return default_arch;
}

typedef struct _KeyFileNewFromFile
{
  GFile *file;
  GKeyFileFlags flags;
} KeyFileNewFromFile;

static void
key_file_new_from_file_free (KeyFileNewFromFile *state)
{
  g_clear_object (&state->file);
  g_free (state);
}

static DexFuture *
foundry_key_file_new_from_file_fiber (gpointer user_data)
{
  KeyFileNewFromFile *state = user_data;
  g_autoptr(GKeyFile) key_file = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GBytes) bytes = NULL;

  g_assert (state != NULL);
  g_assert (G_IS_FILE (state->file));

  key_file = g_key_file_new ();

  if (!(bytes = dex_await_boxed (dex_file_load_contents_bytes (state->file), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!g_key_file_load_from_bytes (key_file, bytes, state->flags, &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_take_boxed (G_TYPE_KEY_FILE, g_steal_pointer (&key_file));
}

/**
 * foundry_key_file_new_from_file:
 * @file: a [iface@Gio.File]
 * @flags: flags that may affect loading
 *
 * Similar to calling g_key_file_new() followed by a load function. This
 * handles both construction and loading as well as doing parsing off
 * of the main thread.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [struct@GLib.KeyFile] or rejects with error
 */
DexFuture *
foundry_key_file_new_from_file (GFile         *file,
                                GKeyFileFlags  flags)
{
  KeyFileNewFromFile *state;

  dex_return_error_if_fail (G_IS_FILE (file));

  state = g_new0 (KeyFileNewFromFile, 1);
  state->file = g_object_ref (file);
  state->flags = flags;

  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              foundry_key_file_new_from_file_fiber,
                              state,
                              (GDestroyNotify) key_file_new_from_file_free);
}

typedef struct _FileTest
{
  DexPromise *promise;
  char *path;
  GFileTest test;
} FileTest;

static void
foundry_file_test_func (gpointer data)
{
  FileTest *state = data;

  dex_promise_resolve_boolean (state->promise,
                               g_file_test (state->path, state->test));

  dex_clear (&state->promise);
  g_free (state->path);
  g_free (state);
}

/**
 * foundry_file_test:
 * @path: the path to check
 * @test: the #GFileTest to check for
 *
 * Similar to g_file_test() but performed on the thread pool and yields a future.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a boolean
 *   of %TRUE if the test was met, otherwise resolves to %FALSE.
 */
DexFuture *
foundry_file_test (const char *path,
                   GFileTest   test)
{
  FileTest *state = g_new0 (FileTest, 1);
  DexPromise *promise = dex_promise_new ();

  state->path = g_strdup (path);
  state->test = test;
  state->promise = dex_ref (promise);

  dex_scheduler_push (dex_thread_pool_scheduler_get_default (),
                      foundry_file_test_func,
                      state);

  return DEX_FUTURE (promise);
}

char *
foundry_dup_projects_directory (void)
{
  g_autoptr(GSettings) settings = g_settings_new ("app.devsuite.foundry");
  g_autofree char *projects_directory = g_settings_get_string (settings, "projects-directory");

  if (foundry_str_empty0 (projects_directory))
    {
      g_clear_pointer (&projects_directory, g_free);
      projects_directory = g_build_filename (g_get_home_dir (), _("Projects"), NULL);
    }

  foundry_path_expand_inplace (&projects_directory);

  return g_steal_pointer (&projects_directory);
}

/**
 * foundry_dup_projects_directory_file:
 *
 * Returns: (transfer full):
 */
GFile *
foundry_dup_projects_directory_file (void)
{
  g_autofree char *path = foundry_dup_projects_directory ();

  return g_file_new_for_path (path);
}

typedef struct _KeyFileMerged
{
  char **search_dirs;
  char *file;
  GKeyFileFlags flags;
} KeyFileMerged;

static void
key_file_merged_free (KeyFileMerged *state)
{
  g_clear_pointer (&state->search_dirs, g_strfreev);
  g_clear_pointer (&state->file, g_free);
  g_free (state);
}

static void
merge_down (GKeyFile *base,
            GKeyFile *upper)
{
  g_auto(GStrv) groups = g_key_file_get_groups (upper, NULL);

  for (gsize g = 0; groups[g]; g++)
    {
      g_auto(GStrv) keys = g_key_file_get_keys (upper, groups[g], NULL, NULL);

      for (gsize k = 0; keys[k]; k++)
        {
          g_autofree char *value = g_key_file_get_value (upper, groups[g], keys[k], NULL);
          g_key_file_set_value (base, groups[g], keys[k], value);
        }
    }
}

static DexFuture *
foundry_key_file_new_merged_fiber (gpointer data)
{
  KeyFileMerged *state = data;
  g_autoptr(GKeyFile) key_file = NULL;
  gsize n_search_dirs;
  gsize matched = 0;

  g_assert (state != NULL);
  g_assert (state->search_dirs != NULL);
  g_assert (state->file != NULL);

  n_search_dirs = g_strv_length (state->search_dirs);
  key_file = g_key_file_new ();

  for (gsize i = 0; i < n_search_dirs; i++)
    {
      const char *search_dir = state->search_dirs[i];
      g_autoptr(GFile) file = g_file_new_build_filename (search_dir, state->file, NULL);
      g_autoptr(GKeyFile) layer = NULL;
      g_autoptr(GError) error = NULL;

      if ((layer = dex_await_boxed (foundry_key_file_new_from_file (file, state->flags), &error)))
        {
          merge_down (key_file, layer);
          matched++;
        }
    }

  if (matched == 0)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_FOUND,
                                  "File \"%s\" not found in search dirs",
                                  state->file);

  return dex_future_new_take_boxed (G_TYPE_KEY_FILE, g_steal_pointer (&key_file));
}

/**
 * foundry_key_file_new_merged:
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [struct@GLib.KeyFile] or rejects with error.
 */
DexFuture *
foundry_key_file_new_merged (const char * const *search_dirs,
                             const char         *file,
                             GKeyFileFlags       flags)
{
  KeyFileMerged *state;

  dex_return_error_if_fail (search_dirs != NULL);
  dex_return_error_if_fail (file != NULL);

  state = g_new0 (KeyFileMerged, 1);
  state->search_dirs = g_strdupv ((char **)search_dirs);
  state->file = g_strdup (file);
  state->flags = flags;

  return dex_scheduler_spawn (NULL, 0,
                              foundry_key_file_new_merged_fiber,
                              state,
                              (GDestroyNotify)key_file_merged_free);
}

char *
_foundry_get_shared_dir (void)
{
  g_autoptr(GSettings) settings = g_settings_new ("app.devsuite.foundry");
  g_autofree char *shared_data_dir = g_settings_get_string (settings, "shared-data-directory");

  if (shared_data_dir[0] == 0)
    {
      g_clear_pointer (&shared_data_dir, g_free);
      shared_data_dir = g_build_filename (g_get_user_config_dir (), "foundry", "shared", NULL);
    }

  foundry_path_expand_inplace (&shared_data_dir);

  return g_steal_pointer (&shared_data_dir);
}

const char *
foundry_get_version_string (void)
{
  static const char *version = FOUNDRY_VERSION_S;
  return version;
}

static void
_foundry_write_all_bytes_cb (GObject      *object,
                             GAsyncResult *result,
                             gpointer      user_data)
{
  gpointer *state = user_data;
  g_autoptr(GError) error = NULL;
  gsize n_written = 0;

  g_assert (G_IS_OUTPUT_STREAM (object));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (state != NULL);
  g_assert (state[0] != NULL);
  g_assert (DEX_IS_PROMISE (state[1]));

  if (!g_output_stream_writev_all_finish (G_OUTPUT_STREAM (object), result, &n_written, &error))
    dex_promise_reject (state[1], g_steal_pointer (&error));
  else
    dex_promise_resolve_int64 (state[1], n_written);

  g_ptr_array_unref (state[0]);
  dex_unref (state[1]);
  g_array_unref (state[2]);
  g_free (state);
}

DexFuture *
_foundry_write_all_bytes (GOutputStream  *stream,
                          GBytes        **bytesv,
                          guint           n_bytesv)
{
  DexPromise *promise;
  g_autoptr(GPtrArray) ar = NULL;
  g_autoptr(GArray) vec = NULL;
  gpointer *state;

  dex_return_error_if_fail (G_IS_OUTPUT_STREAM (stream));
  dex_return_error_if_fail (bytesv != NULL);
  dex_return_error_if_fail (n_bytesv > 0);

  promise = dex_promise_new_cancellable ();
  vec = g_array_new (FALSE, FALSE, sizeof (GOutputVector));
  ar = g_ptr_array_new_with_free_func ((GDestroyNotify)g_bytes_unref);

  for (guint i = 0; i < n_bytesv; i++)
    {
      GBytes *bytes = bytesv[i];
      GOutputVector ov;

      ov.buffer = g_bytes_get_data (bytes, NULL);
      ov.size = g_bytes_get_size (bytes);

      g_ptr_array_add (ar, g_bytes_ref (bytes));
      g_array_append_val (vec, ov);
    }

  state = g_new0 (gpointer, 3);
  state[0] = g_ptr_array_ref (ar);
  state[1] = dex_ref (promise);
  state[2] = g_array_ref (vec);

  g_output_stream_writev_all_async (stream,
                                    (GOutputVector *)(gpointer)vec->data,
                                    vec->len,
                                    G_PRIORITY_DEFAULT,
                                    dex_promise_get_cancellable (promise),
                                    _foundry_write_all_bytes_cb,
                                    state);

  return DEX_FUTURE (promise);
}

#ifndef HAVE_PIPE2
static int
pipe2 (int      fd_pair[2],
       unsigned flags)
{
  int r = pipe (fd_pair);
  int errsv;

  if (r != 0)
    return r;

  if (flags & O_CLOEXEC)
    {
      if (fcntl (fd_pair[0], F_SETFD, FD_CLOEXEC) != 0)
        goto failure;

      if (fcntl (fd_pair[1], F_SETFD, FD_CLOEXEC) != 0)
        goto failure;
    }

  if (flags & O_NONBLOCK)
    {
      if (fcntl (fd_pair[0], F_SETFD, FD_NONBLOCK) != 0)
        goto failure;

      if (fcntl (fd_pair[1], F_SETFD, FD_NONBLOCK) != 0)
        goto failure;
    }

  return 0;

failure:
  errsv = errno;

  close (fd_pair[0]);
  close (fd_pair[1]);

  fd_pair[0] = -1;
  fd_pair[1] = -1;

  errno = errsv;
}
#endif


gboolean
foundry_pipe (int     *read_fd,
              int     *write_fd,
              int      flags,
              GError **error)
{
  int fds[2];

  if (pipe2 (fds, flags) != 0)
    {
      int errsv = errno;
      g_set_error_literal (error,
                           G_IO_ERROR,
                           g_io_error_from_errno (errsv),
                           g_strerror (errsv));
      return FALSE;
    }

  *read_fd = fds[0];
  *write_fd = fds[1];

  return TRUE;
}

static DexFuture *
add_to_store (DexFuture *completed,
              gpointer   user_data)
{
  g_autoptr(GListModel) model = dex_await_object (dex_ref (completed), NULL);
  GListStore *store = user_data;

  g_assert (G_IS_LIST_STORE (store));
  g_assert (!model || G_IS_LIST_MODEL (model));

  if (model != NULL)
    g_list_store_append (store, model);

  return dex_future_new_true ();
}

DexFuture *
_foundry_flatten_list_model_new_from_futures (GPtrArray *array)
{
  g_autoptr(GListStore) store = g_list_store_new (G_TYPE_LIST_MODEL);
  g_autoptr(GListModel) flatten = NULL;
  g_autoptr(GPtrArray) all = NULL;

  flatten = foundry_flatten_list_model_new (g_object_ref (G_LIST_MODEL (store)));
  all = g_ptr_array_new_with_free_func (dex_unref);

  if (array != NULL && array->len > 0)
    {
      g_autoptr(DexFuture) future = NULL;

      for (guint i = 0; i < array->len; i++)
        g_ptr_array_add (all,
                         dex_future_finally (dex_ref (g_ptr_array_index (array, i)),
                                             add_to_store,
                                             g_object_ref (store),
                                             g_object_unref));

      future = dex_future_catch (foundry_future_all (all),
                                 foundry_future_return_true,
                                 NULL, NULL);

      foundry_list_model_set_future (G_LIST_MODEL (flatten), future);
    }

  return dex_future_new_take_object (g_steal_pointer (&flatten));
}
