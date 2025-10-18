/* foundry-unix-fd-map.c
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
#include <fcntl.h>
#include <unistd.h>

#include <glib/gstdio.h>

#include <glib-unix.h>

#include <gio/gunixinputstream.h>
#include <gio/gunixoutputstream.h>

#include "foundry-debug.h"
#include "foundry-unix-fd-map.h"
#include "foundry-util.h"

typedef struct
{
  int source_fd;
  int dest_fd;
} FoundryUnixFDMapItem;

struct _FoundryUnixFDMap
{
  GObject  parent_instance;
  GArray  *map;
};

G_DEFINE_FINAL_TYPE (FoundryUnixFDMap, foundry_unix_fd_map, G_TYPE_OBJECT)

static void
item_clear (gpointer data)
{
  FoundryUnixFDMapItem *item = data;

  item->dest_fd = -1;

  if (item->source_fd != -1)
    {
      close (item->source_fd);
      item->source_fd = -1;
    }
}

static void
foundry_unix_fd_map_dispose (GObject *object)
{
  FoundryUnixFDMap *self = (FoundryUnixFDMap *)object;

  if (self->map->len > 0)
    g_array_remove_range (self->map, 0, self->map->len);

  G_OBJECT_CLASS (foundry_unix_fd_map_parent_class)->dispose (object);
}

static void
foundry_unix_fd_map_finalize (GObject *object)
{
  FoundryUnixFDMap *self = (FoundryUnixFDMap *)object;

  g_clear_pointer (&self->map, g_array_unref);

  G_OBJECT_CLASS (foundry_unix_fd_map_parent_class)->finalize (object);
}

static void
foundry_unix_fd_map_class_init (FoundryUnixFDMapClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_unix_fd_map_dispose;
  object_class->finalize = foundry_unix_fd_map_finalize;
}

static void
foundry_unix_fd_map_init (FoundryUnixFDMap *self)
{
  self->map = g_array_new (FALSE, FALSE, sizeof (FoundryUnixFDMapItem));
  g_array_set_clear_func (self->map, item_clear);
}

FoundryUnixFDMap *
foundry_unix_fd_map_new (void)
{
  return g_object_new (FOUNDRY_TYPE_UNIX_FD_MAP, NULL);
}

guint
foundry_unix_fd_map_get_length (FoundryUnixFDMap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), 0);

  return self->map->len;
}

void
foundry_unix_fd_map_take (FoundryUnixFDMap *self,
                          int               source_fd,
                          int               dest_fd)
{
  FoundryUnixFDMapItem insert;

  g_return_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self));
  g_return_if_fail (dest_fd > -1);

  for (guint i = 0; i < self->map->len; i++)
    {
      FoundryUnixFDMapItem *item = &g_array_index (self->map, FoundryUnixFDMapItem, i);

      if (item->dest_fd == dest_fd)
        {
          if (item->source_fd != -1)
            close (item->source_fd);
          item->source_fd = source_fd;
          return;
        }
    }

  insert.source_fd = source_fd;
  insert.dest_fd = dest_fd;
  g_array_append_val (self->map, insert);
}

int
foundry_unix_fd_map_steal (FoundryUnixFDMap *self,
                           guint             index,
                           int              *dest_fd)
{
  FoundryUnixFDMapItem *steal;

  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), -1);
  g_return_val_if_fail (index < self->map->len, -1);

  steal = &g_array_index (self->map, FoundryUnixFDMapItem, index);

  if (dest_fd != NULL)
    *dest_fd = steal->dest_fd;

  return g_steal_fd (&steal->source_fd);
}

int
foundry_unix_fd_map_get (FoundryUnixFDMap  *self,
                         guint              index,
                         int               *dest_fd,
                         GError           **error)
{
  FoundryUnixFDMapItem *item;
  int ret = -1;

  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), -1);
  g_return_val_if_fail (index < self->map->len, -1);

  item = &g_array_index (self->map, FoundryUnixFDMapItem, index);

  if (item->source_fd == -1)
    {
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_CLOSED,
                   "File-descriptor at index %u already stolen",
                   index);
      return -1;
    }

  ret = dup (item->source_fd);

  if (ret == -1)
    {
      int errsv = errno;
      g_set_error_literal (error,
                           G_IO_ERROR,
                           g_io_error_from_errno (errsv),
                           g_strerror (errsv));
      return -1;
    }

  return ret;
}

int
foundry_unix_fd_map_peek (FoundryUnixFDMap *self,
                          guint             index,
                          int              *dest_fd)
{
  const FoundryUnixFDMapItem *item;

  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), -1);
  g_return_val_if_fail (index < self->map->len, -1);

  item = &g_array_index (self->map, FoundryUnixFDMapItem, index);

  if (dest_fd != NULL)
    *dest_fd = item->dest_fd;

  return item->source_fd;
}

static int
foundry_unix_fd_map_peek_for_dest_fd (FoundryUnixFDMap *self,
                                      int               dest_fd)
{
  g_assert (FOUNDRY_IS_UNIX_FD_MAP (self));
  g_assert (dest_fd != -1);

  for (guint i = 0; i < self->map->len; i++)
    {
      const FoundryUnixFDMapItem *item = &g_array_index (self->map, FoundryUnixFDMapItem, i);

      if (item->dest_fd == dest_fd)
        return item->source_fd;
    }

  return -1;
}

int
foundry_unix_fd_map_peek_stdin (FoundryUnixFDMap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), -1);

  return foundry_unix_fd_map_peek_for_dest_fd (self, STDIN_FILENO);
}

int
foundry_unix_fd_map_peek_stdout (FoundryUnixFDMap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), -1);

  return foundry_unix_fd_map_peek_for_dest_fd (self, STDOUT_FILENO);
}

int
foundry_unix_fd_map_peek_stderr (FoundryUnixFDMap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), -1);

  return foundry_unix_fd_map_peek_for_dest_fd (self, STDERR_FILENO);
}

static int
foundry_unix_fd_map_steal_for_dest_fd (FoundryUnixFDMap *self,
                                       int               dest_fd)
{
  g_assert (FOUNDRY_IS_UNIX_FD_MAP (self));
  g_assert (dest_fd != -1);

  for (guint i = 0; i < self->map->len; i++)
    {
      FoundryUnixFDMapItem *item = &g_array_index (self->map, FoundryUnixFDMapItem, i);

      if (item->dest_fd == dest_fd)
        return g_steal_fd (&item->source_fd);
    }

  return -1;
}

int
foundry_unix_fd_map_steal_stdin (FoundryUnixFDMap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), -1);

  return foundry_unix_fd_map_steal_for_dest_fd (self, STDIN_FILENO);
}

int
foundry_unix_fd_map_steal_stdout (FoundryUnixFDMap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), -1);

  return foundry_unix_fd_map_steal_for_dest_fd (self, STDOUT_FILENO);
}

int
foundry_unix_fd_map_steal_stderr (FoundryUnixFDMap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), -1);

  return foundry_unix_fd_map_steal_for_dest_fd (self, STDERR_FILENO);
}

static gboolean
foundry_unix_fd_map_isatty (FoundryUnixFDMap *self,
                            int               dest_fd)
{
  g_assert (FOUNDRY_IS_UNIX_FD_MAP (self));
  g_assert (dest_fd != -1);

  for (guint i = 0; i < self->map->len; i++)
    {
      const FoundryUnixFDMapItem *item = &g_array_index (self->map, FoundryUnixFDMapItem, i);

      if (item->dest_fd == dest_fd)
        return item->source_fd != -1 && isatty (item->source_fd);
    }

  return FALSE;
}

gboolean
foundry_unix_fd_map_stdin_isatty (FoundryUnixFDMap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), FALSE);

  return foundry_unix_fd_map_isatty (self, STDIN_FILENO);
}

gboolean
foundry_unix_fd_map_stdout_isatty (FoundryUnixFDMap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), FALSE);

  return foundry_unix_fd_map_isatty (self, STDOUT_FILENO);
}

gboolean
foundry_unix_fd_map_stderr_isatty (FoundryUnixFDMap *self)
{
  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), FALSE);

  return foundry_unix_fd_map_isatty (self, STDERR_FILENO);
}

int
foundry_unix_fd_map_get_max_dest_fd (FoundryUnixFDMap *self)
{
  int max_dest_fd = 2;

  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), -1);

  for (guint i = 0; i < self->map->len; i++)
    {
      const FoundryUnixFDMapItem *item = &g_array_index (self->map, FoundryUnixFDMapItem, i);

      if (item->dest_fd > max_dest_fd)
        max_dest_fd = item->dest_fd;
    }

  return max_dest_fd;
}

gboolean
foundry_unix_fd_map_open_file (FoundryUnixFDMap  *self,
                               const char        *filename,
                               int                dest_fd,
                               int                mode,
                               GError           **error)
{
  int fd;

  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), FALSE);
  g_return_val_if_fail (filename != NULL, FALSE);
  g_return_val_if_fail (dest_fd > -1, FALSE);

  if (-1 == (fd = open (filename, mode)))
    {
      int errsv = errno;
      g_set_error_literal (error,
                           G_IO_ERROR,
                           g_io_error_from_errno (errsv),
                           g_strerror (errsv));
      return FALSE;
    }

  foundry_unix_fd_map_take (self, fd, dest_fd);

  return TRUE;
}

gboolean
foundry_unix_fd_map_steal_from (FoundryUnixFDMap  *self,
                                FoundryUnixFDMap  *other,
                                GError           **error)
{
  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), FALSE);
  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (other), FALSE);

  for (guint i = 0; i < other->map->len; i++)
    {
      FoundryUnixFDMapItem *item = &g_array_index (other->map, FoundryUnixFDMapItem, i);

      if (item->source_fd != -1)
        {
          for (guint j = 0; j < self->map->len; j++)
            {
              FoundryUnixFDMapItem *ele = &g_array_index (self->map, FoundryUnixFDMapItem, j);

              if (ele->dest_fd == item->dest_fd && ele->source_fd != -1)
                {
                  g_set_error (error,
                               G_IO_ERROR,
                               G_IO_ERROR_INVALID_ARGUMENT,
                               "Attempt to merge overlapping destination FDs for %d",
                               item->dest_fd);
                  return FALSE;
                }
            }
        }

      foundry_unix_fd_map_take (self, g_steal_fd (&item->source_fd), item->dest_fd);
    }

  return TRUE;
}

/**
 * foundry_unix_fd_map_create_stream:
 * @self: a #FoundryUnixFDMap
 * @dest_read_fd: the FD number in the destination process for the read side (stdin)
 * @dest_write_fd: the FD number in the destination process for the write side (stdout)
 *
 * Creates a #GIOStream to communicate with another process.
 *
 * Use this to create a #GIOStream to use from the calling process to communicate
 * with a subprocess. Generally, you should pass STDIN_FILENO for @dest_read_fd
 * and STDOUT_FILENO for @dest_write_fd.
 *
 * Returns: (transfer full): a #GIOStream if successful; otherwise %NULL and
 *   @error is set.
 */
GIOStream *
foundry_unix_fd_map_create_stream (FoundryUnixFDMap  *self,
                                   int                dest_read_fd,
                                   int                dest_write_fd,
                                   GError           **error)
{
  g_autoptr(GIOStream) ret = NULL;
  g_autoptr(GInputStream) input = NULL;
  g_autoptr(GOutputStream) output = NULL;
  int stdin_pair[2] = {-1,-1};
  int stdout_pair[2] = {-1,-1};

  FOUNDRY_ENTRY;

  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), NULL);
  g_return_val_if_fail (dest_read_fd > -1, NULL);
  g_return_val_if_fail (dest_write_fd > -1, NULL);

  if (!foundry_pipe (&stdin_pair[0], &stdin_pair[1], O_CLOEXEC, error) ||
      !foundry_pipe (&stdout_pair[0], &stdout_pair[1], O_CLOEXEC, error))
    FOUNDRY_GOTO (failure);

  g_assert (stdin_pair[0] != -1);
  g_assert (stdin_pair[1] != -1);
  g_assert (stdout_pair[0] != -1);
  g_assert (stdout_pair[1] != -1);

  foundry_unix_fd_map_take (self, g_steal_fd (&stdin_pair[0]), dest_read_fd);
  foundry_unix_fd_map_take (self, g_steal_fd (&stdout_pair[1]), dest_write_fd);

  if (!g_unix_set_fd_nonblocking (stdin_pair[1], TRUE, error) ||
      !g_unix_set_fd_nonblocking (stdout_pair[0], TRUE, error))
    FOUNDRY_GOTO (failure);

  output = g_unix_output_stream_new (g_steal_fd (&stdin_pair[1]), TRUE);
  input = g_unix_input_stream_new (g_steal_fd (&stdout_pair[0]), TRUE);

  ret = g_simple_io_stream_new (input, output);

  g_assert (stdin_pair[0] == -1);
  g_assert (stdin_pair[1] == -1);
  g_assert (stdout_pair[0] == -1);
  g_assert (stdout_pair[1] == -1);

failure:

  if (stdin_pair[0] != -1)
    close (stdin_pair[0]);
  if (stdin_pair[1] != -1)
    close (stdin_pair[1]);
  if (stdout_pair[0] != -1)
    close (stdout_pair[0]);
  if (stdout_pair[1] != -1)
    close (stdout_pair[1]);

  FOUNDRY_RETURN (g_steal_pointer (&ret));
}

gboolean
foundry_unix_fd_map_silence_fd (FoundryUnixFDMap  *self,
                                int                dest_fd,
                                GError           **error)
{
  int null_fd = -1;

  g_return_val_if_fail (FOUNDRY_IS_UNIX_FD_MAP (self), FALSE);

  if (dest_fd < 0)
    return TRUE;

  if (-1 == (null_fd = open ("/dev/null", O_WRONLY)))
    {
      int errsv = errno;
      g_set_error_literal (error,
                           G_IO_ERROR,
                           g_io_error_from_errno (errsv),
                           g_strerror (errsv));
      return FALSE;
    }

  foundry_unix_fd_map_take (self, g_steal_fd (&null_fd), dest_fd);

  return TRUE;
}
