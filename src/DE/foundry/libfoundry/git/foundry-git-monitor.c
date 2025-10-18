/* foundry-git-monitor.c
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

#include "foundry-git-monitor-private.h"

#define DELAY_MSEC 250

struct _FoundryGitMonitor
{
  GObject       parent_instance;
  char         *git_dir;
  GFileMonitor *dotgit_monitor;
  GFileMonitor *dotgit_logs_monitor;
  GFileMonitor *dotgit_refs_heads_monitor;
  DexPromise   *when_changed;
  guint         changed_source;
};

G_DEFINE_FINAL_TYPE (FoundryGitMonitor, foundry_git_monitor, G_TYPE_OBJECT)

enum {
  CHANGED,
  N_SIGNALS
};

static guint signals[N_SIGNALS];

static void
foundry_git_monitor_finalize (GObject *object)
{
  FoundryGitMonitor *self = (FoundryGitMonitor *)object;

  if (self->when_changed != NULL)
    {
      g_autoptr(DexPromise) when_changed = g_steal_pointer (&self->when_changed);

      dex_promise_reject (when_changed,
                          g_error_new (G_IO_ERROR,
                                       G_IO_ERROR_CANCELLED,
                                       "Monitor disposed"));
    }

  g_clear_handle_id (&self->changed_source, g_source_remove);

  g_clear_pointer (&self->git_dir, g_free);

  g_clear_object (&self->dotgit_monitor);
  g_clear_object (&self->dotgit_logs_monitor);
  g_clear_object (&self->dotgit_refs_heads_monitor);

  G_OBJECT_CLASS (foundry_git_monitor_parent_class)->finalize (object);
}

static void
foundry_git_monitor_class_init (FoundryGitMonitorClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_git_monitor_finalize;

  signals[CHANGED] =
    g_signal_new ("changed",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL, NULL,
                  NULL,
                  G_TYPE_NONE, 0);
}

static void
foundry_git_monitor_init (FoundryGitMonitor *self)
{
}

static gboolean
foundry_git_monitor_emit_changed (gpointer data)
{
  FoundryGitMonitor *self = data;

  g_assert (FOUNDRY_IS_GIT_MONITOR (self));

  self->changed_source = 0;

  if (self->when_changed != NULL)
    {
      g_autoptr(DexPromise) when_changed = g_steal_pointer (&self->when_changed);

      dex_promise_resolve_boolean (when_changed, TRUE);
    }

  g_signal_emit (self, signals[CHANGED], 0);

  return G_SOURCE_REMOVE;
}

static void
foundry_git_monitor_queue_changed (FoundryGitMonitor *self)
{
  g_assert (FOUNDRY_IS_GIT_MONITOR (self));

  if (self->changed_source == 0)
    self->changed_source = g_timeout_add (DELAY_MSEC, foundry_git_monitor_emit_changed, self);
}

static void
foundry_git_monitor_dotgit_cb (FoundryGitMonitor *self,
                               GFile             *file,
                               GFile             *other_file,
                               GFileMonitorEvent  event,
                               GFileMonitor      *monitor)
{
  g_autofree char *name = NULL;

  g_assert (FOUNDRY_IS_GIT_MONITOR (self));
  g_assert (G_IS_FILE (file));
  g_assert (!other_file || G_IS_FILE (other_file));
  g_assert (G_IS_FILE_MONITOR (monitor));

  name = g_file_get_basename (file);

  if (g_str_equal (name, "HEAD") ||
      g_str_equal (name, "index") ||
      g_str_equal (name, "logs") ||
      g_str_equal (name, "packed-refs"))
    foundry_git_monitor_queue_changed (self);
}

static void
foundry_git_monitor_dotgit_logs_cb (FoundryGitMonitor *self,
                                    GFile             *file,
                                    GFile             *other_file,
                                    GFileMonitorEvent  event,
                                    GFileMonitor      *monitor)
{
  g_autofree char *name = NULL;

  g_assert (FOUNDRY_IS_GIT_MONITOR (self));
  g_assert (G_IS_FILE (file));
  g_assert (!other_file || G_IS_FILE (other_file));
  g_assert (G_IS_FILE_MONITOR (monitor));

  name = g_file_get_basename (file);

  if (g_str_equal (name, "logs"))
    foundry_git_monitor_queue_changed (self);
}

static void
foundry_git_monitor_dotgit_refs_heads_cb (FoundryGitMonitor *self,
                                          GFile             *file,
                                          GFile             *other_file,
                                          GFileMonitorEvent  event,
                                          GFileMonitor      *monitor)
{
  g_assert (FOUNDRY_IS_GIT_MONITOR (self));
  g_assert (G_IS_FILE (file));
  g_assert (!other_file || G_IS_FILE (other_file));
  g_assert (G_IS_FILE_MONITOR (monitor));

  foundry_git_monitor_queue_changed (self);
}

static DexFuture *
foundry_git_monitor_new_thread (gpointer data)
{
  const char *git_dir = data;
  g_autoptr(FoundryGitMonitor) self = NULL;
  g_autoptr(GFileMonitor) dotgit_monitor = NULL;
  g_autoptr(GFileMonitor) dotgit_logs_monitor = NULL;
  g_autoptr(GFileMonitor) dotgit_refs_heads_monitor = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) dotgit = NULL;
  g_autoptr(GFile) dotgit_logs = NULL;
  g_autoptr(GFile) dotgit_refs_heads = NULL;

  g_assert (git_dir != NULL);

  self = g_object_new (FOUNDRY_TYPE_GIT_MONITOR, NULL);
  self->git_dir = g_strdup (git_dir);

  dotgit = g_file_new_for_path (git_dir);
  dotgit_logs = g_file_get_child (dotgit, "refs/logs/");
  dotgit_refs_heads = g_file_get_child (dotgit, "refs/heads/");

  if (!(dotgit_monitor = g_file_monitor_directory (dotgit, G_FILE_MONITOR_NONE, NULL, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  g_signal_connect_object (dotgit_monitor,
                           "changed",
                           G_CALLBACK (foundry_git_monitor_dotgit_cb),
                           self,
                           G_CONNECT_SWAPPED);

  if (!(dotgit_refs_heads_monitor = g_file_monitor_directory (dotgit_refs_heads, G_FILE_MONITOR_NONE, NULL, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  g_signal_connect_object (dotgit_refs_heads_monitor,
                           "changed",
                           G_CALLBACK (foundry_git_monitor_dotgit_refs_heads_cb),
                           self,
                           G_CONNECT_SWAPPED);

  if ((dotgit_logs_monitor = g_file_monitor_directory (dotgit_logs, G_FILE_MONITOR_NONE, NULL, &error)))
    g_signal_connect_object (dotgit_logs_monitor,
                             "changed",
                             G_CALLBACK (foundry_git_monitor_dotgit_logs_cb),
                             self,
                             G_CONNECT_SWAPPED);

  self->dotgit_monitor = g_steal_pointer (&dotgit_monitor);
  self->dotgit_logs_monitor = g_steal_pointer (&dotgit_logs_monitor);
  self->dotgit_refs_heads_monitor = g_steal_pointer (&dotgit_refs_heads_monitor);

  return dex_future_new_take_object (g_steal_pointer (&self));
}

DexFuture *
foundry_git_monitor_new (const char  *git_dir)
{
  dex_return_error_if_fail (git_dir != NULL);

  return dex_thread_spawn ("[git-monitor]",
                           foundry_git_monitor_new_thread,
                           g_strdup (git_dir),
                           g_free);
}

DexFuture *
foundry_git_monitor_when_changed (FoundryGitMonitor *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_GIT_MONITOR (self));

  if (self->when_changed == NULL)
    self->when_changed = dex_promise_new ();

  return dex_ref (self->when_changed);
}
