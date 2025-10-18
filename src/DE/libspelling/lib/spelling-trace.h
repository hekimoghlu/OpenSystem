/* spelling-trace.h
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

#pragma once

#include <glib.h>

#ifndef GETTEXT_PACKAGE
# error "config.h was not included before sysprof-trace.h."
#endif

#ifdef HAVE_SYSPROF
# include <sysprof-capture.h>
#endif

G_BEGIN_DECLS

#ifdef HAVE_SYSPROF
# define SPELLING_PROFILER_ENABLED 1
# define SPELLING_PROFILER_CURRENT_TIME SYSPROF_CAPTURE_CURRENT_TIME
# define SPELLING_PROFILER_ACTIVE (sysprof_collector_is_active())
# define SPELLING_PROFILER_BEGIN_MARK gint64 __begin_time = SYSPROF_CAPTURE_CURRENT_TIME
# define SPELLING_PROFILER_END_MARK(name, message) \
  G_STMT_START { \
    gint64 __duration = SYSPROF_CAPTURE_CURRENT_TIME - __begin_time; \
    sysprof_collector_mark (__begin_time, __duration, "Spelling", name, message); \
  } G_STMT_END
# define SPELLING_PROFILER_MARK(duration, name, message) \
  G_STMT_START { \
    sysprof_collector_mark (SYSPROF_CAPTURE_CURRENT_TIME - (duration), \
                            (duration), "Spelling", name, message); \
  } G_STMT_END
# define SPELLING_PROFILER_LOG(format, ...) \
  G_STMT_START { \
    if (SPELLING_PROFILER_ACTIVE) \
      sysprof_collector_log_printf(G_LOG_LEVEL_DEBUG, G_LOG_DOMAIN, format, __VA_ARGS__); \
  } G_STMT_END
#else
# undef SPELLING_PROFILER_ENABLED
# define SPELLING_PROFILER_ACTIVE (0)
# define SPELLING_PROFILER_CURRENT_TIME 0
# define SPELLING_PROFILER_MARK(duration, name, message) G_STMT_START {} G_STMT_END
# define SPELLING_PROFILER_BEGIN_MARK G_STMT_START {} G_STMT_END
# define SPELLING_PROFILER_END_MARK(name, message) G_STMT_START {} G_STMT_END
# define SPELLING_PROFILER_LOG(format, ...) G_STMT_START {} G_STMT_END
#endif

G_END_DECLS
