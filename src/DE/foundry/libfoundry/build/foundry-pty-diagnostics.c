/* foundry-pty-diagnostics.c
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

#include <glib/gstdio.h>

#include "foundry-build-manager.h"
#include "foundry-build-pipeline.h"
#include "foundry-debug.h"
#include "foundry-diagnostic.h"
#include "foundry-diagnostic-builder.h"
#include "foundry-path.h"
#include "foundry-pty-diagnostics.h"

#include "line-reader-private.h"
#include "pty-intercept.h"

struct _FoundryPtyDiagnostics
{
  FoundryContextual  parent_instance;
  GListStore        *diagnostics;
  GFile             *workdir;
  char              *builddir;
  char              *errfmt_current_dir;
  char              *errfmt_top_dir;
  int                pty_fd;
  PtyIntercept       intercept;
};

static GType
foundry_pty_diagnostics_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_DIAGNOSTIC;
}

static guint
foundry_pty_diagnostics_get_n_items (GListModel *model)
{
  return g_list_model_get_n_items (G_LIST_MODEL ((FOUNDRY_PTY_DIAGNOSTICS (model)->diagnostics)));
}

static gpointer
foundry_pty_diagnostics_get_item (GListModel *model,
                                  guint       position)
{
  return g_list_model_get_item (G_LIST_MODEL ((FOUNDRY_PTY_DIAGNOSTICS (model)->diagnostics)), position);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_pty_diagnostics_get_item_type;
  iface->get_n_items = foundry_pty_diagnostics_get_n_items;
  iface->get_item = foundry_pty_diagnostics_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryPtyDiagnostics, foundry_pty_diagnostics, FOUNDRY_TYPE_CONTEXTUAL,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

G_LOCK_DEFINE (all_regexes);
static GPtrArray *all_regexes;
static GHashTable *severities;

static void
foundry_pty_diagnostics_finalize (GObject *object)
{
  FoundryPtyDiagnostics *self = (FoundryPtyDiagnostics *)object;

  g_clear_object (&self->diagnostics);
  g_clear_object (&self->workdir);

  g_clear_pointer (&self->builddir, g_free);
  g_clear_pointer (&self->errfmt_current_dir, g_free);
  g_clear_pointer (&self->errfmt_top_dir, g_free);

  if (IS_PTY_INTERCEPT (&self->intercept))
    pty_intercept_clear (&self->intercept);

  g_clear_fd (&self->pty_fd, NULL);

  G_OBJECT_CLASS (foundry_pty_diagnostics_parent_class)->finalize (object);
}

static void
foundry_pty_diagnostics_class_init (FoundryPtyDiagnosticsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_pty_diagnostics_finalize;

  severities = g_hash_table_new (g_str_hash, g_str_equal);
#define ADD(name, VALUE) \
  g_hash_table_insert(severities, (char*)name, GUINT_TO_POINTER(FOUNDRY_DIAGNOSTIC_##VALUE))
  ADD ("DEPRECATED", DEPRECATED);
  ADD ("Deprecated", DEPRECATED);
  ADD ("deprecated", DEPRECATED);
  ADD ("ERROR", ERROR);
  ADD ("Error", ERROR);
  ADD ("error", ERROR);
  ADD ("FATAL", FATAL);
  ADD ("Fatal", FATAL);
  ADD ("fatal", FATAL);
  ADD ("IGNORED", IGNORED);
  ADD ("Ignored", IGNORED);
  ADD ("ignored", IGNORED);
  ADD ("NOTE", NOTE);
  ADD ("Note", NOTE);
  ADD ("note", NOTE);
  ADD ("UNUSED", UNUSED);
  ADD ("Unused", UNUSED);
  ADD ("unused", UNUSED);
  ADD ("WARNING", WARNING);
  ADD ("Warning", WARNING);
  ADD ("warning", WARNING);
#undef ADD

  /* Arduino */
  foundry_pty_diagnostics_register (g_regex_new ("(?<filename>[a-zA-Z0-9\\-\\.\\/_]+\\.ino):"
                                                 "(?<line>\\d+):"
                                                 "(?<column>\\d+): "
                                                 ".+(?<level>(?:error|warning)): "
                                                 "(?<message>.*)",
                                                 G_REGEX_OPTIMIZE, 0, NULL));

  /* Dub */
  foundry_pty_diagnostics_register (g_regex_new ("(?<filename>[a-zA-Z0-9\\-\\.\\/_]+.d)"
                                                 "(?<line>\\(\\d+),(?<column>\\d+)\\)"
                                                 ": (?<level>.+(?=:))(?<message>.*)",
                                                 G_REGEX_OPTIMIZE, 0, NULL));

  /* GCC */
  foundry_pty_diagnostics_register (g_regex_new ("(?<filename>[a-zA-Z0-9\\+\\-\\.\\/_]+):"
                                                 "(?<line>\\d+):"
                                                 "(?<column>\\d+): "
                                                 "(?<level>[\\w\\s]+): "
                                                 "(?<message>.*)",
                                                 G_REGEX_OPTIMIZE, 0, NULL));

  /* Mono */
  foundry_pty_diagnostics_register (g_regex_new ("(?<filename>[a-zA-Z0-9\\-\\.\\/_]+.cs)"
                                                 "\\((?<line>\\d+),(?<column>\\d+)\\): "
                                                 "(?<level>[\\w\\s]+) "
                                                 "(?<code>CS[0-9]+): "
                                                 "(?<message>.*)",
                                                 G_REGEX_OPTIMIZE, 0, NULL));

  /* Vala */
  foundry_pty_diagnostics_register (g_regex_new ("(?<filename>[a-zA-Z0-9\\-\\.\\/_]+.vala):"
                                                 "(?<line>\\d+).(?<column>\\d+)-(?<line2>\\d+).(?<column2>\\d+): "
                                                 "(?<level>[\\w\\s]+): "
                                                 "(?<message>.*)",
                                                 G_REGEX_OPTIMIZE, 0, NULL));
}

static void
foundry_pty_diagnostics_init (FoundryPtyDiagnostics *self)
{
  self->diagnostics = g_list_store_new (FOUNDRY_TYPE_DIAGNOSTIC);
  self->pty_fd = -1;

  g_signal_connect_object (self->diagnostics,
                           "items-changed",
                           G_CALLBACK (g_list_model_items_changed),
                           self,
                           G_CONNECT_SWAPPED);
}

static gboolean
extract_directory_change (FoundryPtyDiagnostics *self,
                          const guint8          *data,
                          gsize                  len)
{
  g_autofree gchar *dir = NULL;
  const guint8 *begin;

  g_assert (FOUNDRY_IS_PTY_DIAGNOSTICS (self));

  if (len == 0)
    return FALSE;

#define ENTERING_DIRECTORY_BEGIN "Entering directory '"
#define ENTERING_DIRECTORY_END   "'"

  begin = memmem (data, len, ENTERING_DIRECTORY_BEGIN, strlen (ENTERING_DIRECTORY_BEGIN));
  if (begin == NULL)
    return FALSE;

  begin += strlen (ENTERING_DIRECTORY_BEGIN);

  if (data[len - 1] != '\'')
    return FALSE;

  len = &data[len - 1] - begin;
  dir = g_strndup ((gchar *)begin, len);

  if (g_utf8_validate (dir, len, NULL))
    {
      g_free (self->errfmt_current_dir);

      if (len == 0)
        self->errfmt_current_dir = g_strdup (self->errfmt_top_dir);
      else
        self->errfmt_current_dir = g_strndup (dir, len);

      if (self->errfmt_top_dir == NULL)
        self->errfmt_top_dir = g_strdup (self->errfmt_current_dir);

      return TRUE;
    }

#undef ENTERING_DIRECTORY_BEGIN
#undef ENTERING_DIRECTORY_END

  return FALSE;
}

static guint8 *
filter_color_codes (const guint8 *data,
                    gsize         len,
                    gsize        *out_len)
{
  g_autoptr(GByteArray) dst = NULL;

  g_assert (out_len != NULL);

  *out_len = 0;

  if (data == NULL)
    return NULL;
  else if (len == 0)
    return (guint8 *)g_strdup ("");

  dst = g_byte_array_sized_new (len);

  for (gsize i = 0; i < len; i++)
    {
      guint8 ch = data[i];
      guint8 next = (i+1) < len ? data[i+1] : 0;

      if (ch == '\\' && next == 'e')
        {
          i += 2;
        }
      else if (ch == '\033')
        {
          i++;
        }
      else
        {
          g_byte_array_append (dst, &ch, 1);
          continue;
        }

      if (i >= len)
        break;

      if (data[i] == '[')
        i++;

      if (i >= len)
        break;

      for (; i < len; i++)
        {
          ch = data[i];

          if (g_ascii_isdigit (ch) || ch == ' ' || ch == ';')
            continue;

          break;
        }
    }

  *out_len = dst->len;

  return g_byte_array_free (g_steal_pointer (&dst), FALSE);
}

static inline FoundryDiagnosticSeverity
parse_severity (const char *str)
{
  guint value = GPOINTER_TO_UINT (g_hash_table_lookup (severities, str));
  return value ? value : FOUNDRY_DIAGNOSTIC_WARNING;
}

static FoundryDiagnostic *
create_diagnostic (FoundryPtyDiagnostics *self,
                   GMatchInfo            *match_info)
{
  g_autoptr(FoundryDiagnosticBuilder) builder = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autofree char *filename = NULL;
  g_autofree char *line = NULL;
  g_autofree char *column = NULL;
  g_autofree char *message = NULL;
  g_autofree char *level = NULL;
  g_autoptr(GFile) project_dir = NULL;
  g_autoptr(GFile) file = NULL;
  struct {
    gint64 line;
    gint64 column;
    FoundryDiagnosticSeverity severity;
  } parsed = { 0 };

  g_assert (FOUNDRY_IS_PTY_DIAGNOSTICS (self));
  g_assert (match_info != NULL);

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  message = g_match_info_fetch_named (match_info, "message");

  /* XXX: This is a hack to ignore a common but unuseful error message.
   *      This really belongs somewhere else, but it's easier to do the
   *      check here for now. We need proper callback for ErrorRegex in
   *      the future so they can ignore it.
   */
  if (message == NULL || strncmp (message, "#warning _FORTIFY_SOURCE requires compiling with optimization", 61) == 0)
    return NULL;

  filename = g_match_info_fetch_named (match_info, "filename");
  line = g_match_info_fetch_named (match_info, "line");
  column = g_match_info_fetch_named (match_info, "column");
  level = g_match_info_fetch_named (match_info, "level");

  if (line != NULL)
    {
      parsed.line = g_ascii_strtoll (line, NULL, 10);
      if (parsed.line < 1 || parsed.line > G_MAXINT32)
        return NULL;
      parsed.line--;
    }

  if (column != NULL)
    {
      parsed.column = g_ascii_strtoll (column, NULL, 10);
      if (parsed.column < 1 || parsed.column > G_MAXINT32)
        return NULL;
      parsed.column--;
    }

  parsed.severity = parse_severity (level);

  /* Expand local user only, if we get a home-relative path */
  if (filename != NULL && strncmp (filename, "~/", 2) == 0)
    {
      char *expanded = foundry_path_expand (filename);
      g_free (filename);
      filename = expanded;
    }

  if (!g_path_is_absolute (filename))
    {
      char *path;

      if (self->errfmt_current_dir != NULL)
        {
          const char *basedir = self->errfmt_current_dir;

          if (g_str_has_prefix (basedir, self->errfmt_top_dir))
            {
              basedir += strlen (self->errfmt_top_dir);
              if (*basedir == G_DIR_SEPARATOR)
                basedir++;
            }

          path = g_build_filename (basedir, filename, NULL);
          g_free (filename);
          filename = path;
        }
      else if (self->builddir != NULL)
        {
          path = g_build_filename (self->builddir, filename, NULL);
          g_free (filename);
          filename = path;
        }
    }

  project_dir = foundry_context_dup_project_directory (context);

  if (!g_path_is_absolute (filename))
    {
      g_autoptr(GFile) child = g_file_get_child (project_dir, filename);
      char *path = g_file_get_path (child);

      g_free (filename);
      filename = path;
    }

  builder = foundry_diagnostic_builder_new (context);
  foundry_diagnostic_builder_take_message (builder, g_steal_pointer (&message));
  foundry_diagnostic_builder_set_severity (builder, parsed.severity);
  foundry_diagnostic_builder_set_path (builder, filename);
  foundry_diagnostic_builder_set_line (builder, parsed.line);
  foundry_diagnostic_builder_set_line_offset (builder, parsed.column);

  return foundry_diagnostic_builder_end (builder);
}

static void
extract_diagnostics (FoundryPtyDiagnostics *self,
                     const guint8          *data,
                     gsize                  len)
{
  g_autofree guint8 *unescaped = NULL;
  LineReader reader;
  gsize line_len;
  char *line;

  g_assert (FOUNDRY_IS_PTY_DIAGNOSTICS (self));
  g_assert (data != NULL);

  if (len == 0 || all_regexes == NULL)
    return;

  /* If we have any color escape sequences, remove them */
  if G_UNLIKELY (memchr (data, '\033', len) || memmem (data, len, "\\e", 2))
    {
      gsize out_len = 0;

      unescaped = filter_color_codes (data, len, &out_len);

      if (out_len == 0)
        return;

      data = unescaped;
      len = out_len;
    }

  line_reader_init (&reader, (char *)data, len);

  G_LOCK (all_regexes);

  while (NULL != (line = line_reader_next (&reader, &line_len)))
    {
      if (extract_directory_change (self, (const guint8 *)line, line_len))
        continue;

      for (guint i = 0; i < all_regexes->len; i++)
        {
          const GRegex *regex = g_ptr_array_index (all_regexes, i);
          g_autoptr(GMatchInfo) match_info = NULL;

          if (g_regex_match_full (regex, line, line_len, 0, 0, &match_info, NULL))
            {
              g_autoptr(FoundryDiagnostic) diagnostic = create_diagnostic (self, match_info);

              if (diagnostic != NULL)
                {
                  g_list_store_append (self->diagnostics, diagnostic);
                  break;
                }
            }
        }
    }

  G_UNLOCK (all_regexes);
}

static void
intercept_pty_consumer_cb (const PtyIntercept     *intercept,
                           const PtyInterceptSide *side,
                           const guint8           *data,
                           gsize                   len,
                           gpointer                user_data)
{
  FoundryPtyDiagnostics *self = user_data;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (FOUNDRY_IS_PTY_DIAGNOSTICS (self));
  g_assert (intercept != NULL);
  g_assert (side != NULL);
  g_assert (data != NULL);
  g_assert (len > 0);

  extract_diagnostics (self, data, len);
}

static DexFuture *
foundry_pty_diagnostics_pipeline_loaded_cb (DexFuture *completed,
                                            gpointer   user_data)
{
  FoundryPtyDiagnostics *self = user_data;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;

  g_assert (DEX_IS_FUTURE (completed));
  g_assert (FOUNDRY_IS_PTY_DIAGNOSTICS (self));

  if ((pipeline = dex_await_object (dex_ref (completed), NULL)))
    {
      g_free (self->builddir);
      self->builddir = foundry_build_pipeline_dup_builddir (pipeline);
    }

  return dex_future_new_true ();
}

static void
foundry_pty_diagnostics_pipeline_invalidated_cb (FoundryPtyDiagnostics *self,
                                                 FoundryBuildManager   *build_manager)
{
  g_assert (FOUNDRY_IS_PTY_DIAGNOSTICS (self));
  g_assert (FOUNDRY_IS_BUILD_MANAGER (build_manager));

  dex_future_disown (dex_future_then (foundry_build_manager_load_pipeline (build_manager),
                                      foundry_pty_diagnostics_pipeline_loaded_cb,
                                      g_object_ref (self),
                                      g_object_unref));
}

/**
 * foundry_pty_diagnostics_new:
 * @context: a [class@Foundry.Context]
 * @pty_fd: a file-descriptor for PTY consumer
 *
 * @pty_fd should be a valid file-descriptor for the consumer side of the
 * PTY device. This is sometimes, historically, called the "master" side of
 * a PTY device.
 *
 * @pty_fd will be `dup()`d.
 *
 * You can use [method@Foundry.PtyDiagnostics.get_fd] to get a new synthetic
 * PTY consumer for which you can create a producer for and attach to
 * child processes.
 *
 * The resulting [class@Foundry.PtyDiagnostics] is also a
 * [iface@Gio.ListModel] of [class@Foundry.Diagnostic] so you may use the
 * list model API to access diagnostics as they are discovered.
 *
 * Returns: (transfer full):
 */
FoundryPtyDiagnostics *
foundry_pty_diagnostics_new (FoundryContext *context,
                             int             pty_fd)
{
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  FoundryPtyDiagnostics *self;

  g_return_val_if_fail (pty_fd > -1, NULL);

  if (-1 == (pty_fd = dup (pty_fd)))
    return NULL;

  self = g_object_new (FOUNDRY_TYPE_PTY_DIAGNOSTICS,
                       "context", context,
                       NULL);

  self->pty_fd = pty_fd;

  pty_intercept_init (&self->intercept, self->pty_fd, NULL);

  pty_intercept_set_callback (&self->intercept,
                              &self->intercept.consumer,
                              intercept_pty_consumer_cb,
                              self);

  build_manager = foundry_context_dup_build_manager (context);

  g_signal_connect_object (build_manager,
                           "pipeline-invalidated",
                           G_CALLBACK (foundry_pty_diagnostics_pipeline_invalidated_cb),
                           self,
                           G_CONNECT_SWAPPED);

  return self;
}

int
foundry_pty_diagnostics_get_fd (FoundryPtyDiagnostics *self)
{
  g_return_val_if_fail (FOUNDRY_IS_PTY_DIAGNOSTICS (self), -1);
  g_return_val_if_fail (IS_PTY_INTERCEPT (&self->intercept), -1);

  return pty_intercept_get_fd (&self->intercept);
}

void
foundry_pty_diagnostics_reset (FoundryPtyDiagnostics *self)
{
  g_return_if_fail (FOUNDRY_IS_MAIN_THREAD ());
  g_return_if_fail (FOUNDRY_IS_PTY_DIAGNOSTICS (self));

  g_list_store_remove_all (self->diagnostics);
}

/**
 * foundry_pty_diagnostics_register:
 * @regex: (transfer full):
 *
 * Registers @regex for use when extracting diagnostics from a PTY stream.
 *
 * The regex should capture values into named groups that the
 * [class@Foundry.PtyDiagnostics] can turn into [class@Foundry.Diagnostic].
 *
 * Named groups supported are:
 *
 *  - message
 *  - filename
 *  - line
 *  - column
 *  - level
 */
void
foundry_pty_diagnostics_register (GRegex *regex)
{
  G_LOCK (all_regexes);

  if (all_regexes == NULL)
    all_regexes = g_ptr_array_new_with_free_func ((GDestroyNotify) g_regex_unref);

  g_ptr_array_add (all_regexes, g_steal_pointer (&regex));

  G_UNLOCK (all_regexes);
}

int
foundry_pty_diagnostics_create_producer (FoundryPtyDiagnostics  *self,
                                         GError                **error)
{
  int fd;

  g_return_val_if_fail (FOUNDRY_IS_PTY_DIAGNOSTICS (self), -1);

  if (-1 == (fd = pty_intercept_create_producer (foundry_pty_diagnostics_get_fd (self), TRUE)))
    {
      int errsv = errno;
      g_set_error_literal (error,
                           G_IO_ERROR,
                           g_io_error_from_errno (errsv),
                           g_strerror (errsv));
    }

  return fd;
}
