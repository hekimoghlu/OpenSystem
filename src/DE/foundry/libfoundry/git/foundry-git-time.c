/* foundry-git-time.c
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

#include "foundry-git-time.h"

GDateTime *
foundry_git_time_to_date_time (const git_time *when)
{
  g_autoptr(GDateTime) utc = NULL;
  g_autoptr(GTimeZone) tz = NULL;

  g_return_val_if_fail (when != NULL, NULL);

  if (!(utc = g_date_time_new_from_unix_utc (when->time)))
    return NULL;

  /* when->offset is in minutes, GTimeZone wants seconds but
   * it must be < 24 hours to be valid.
   */
  if (((gint64)when->offset * 60) > (60 * 60 * 24))
    return NULL;

  if (!(tz = g_time_zone_new_offset (when->offset * 60)))
    return g_steal_pointer (&utc);

  return g_date_time_to_timezone (utc, tz);
}
