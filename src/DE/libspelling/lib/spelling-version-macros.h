/*
 * spelling-version-macros.h
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
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

#if !defined(LIBSPELLING_INSIDE) && !defined(LIBSPELLING_COMPILATION)
# error "Only <libspelling.h> can be included directly."
#endif

#include <glib.h>

#include "spelling-version.h"

#ifndef _SPELLING_EXTERN
# define _SPELLING_EXTERN extern
#endif

#define SPELLING_VERSION_CUR_STABLE (G_ENCODE_VERSION (SPELLING_MAJOR_VERSION, 0))

#ifdef SPELLING_DISABLE_DEPRECATION_WARNINGS
# define SPELLING_DEPRECATED _SPELLING_EXTERN
# define SPELLING_DEPRECATED_FOR(f) _SPELLING_EXTERN
# define SPELLING_UNAVAILABLE(maj,min) _SPELLING_EXTERN
#else
# define SPELLING_DEPRECATED G_DEPRECATED _SPELLING_EXTERN
# define SPELLING_DEPRECATED_FOR(f) G_DEPRECATED_FOR (f) _SPELLING_EXTERN
# define SPELLING_UNAVAILABLE(maj,min) G_UNAVAILABLE (maj, min) _SPELLING_EXTERN
#endif

#define SPELLING_VERSION_1_0 (G_ENCODE_VERSION (1, 0))

#if SPELLING_MAJOR_VERSION == SPELLING_VERSION_1_0
# define SPELLING_VERSION_PREV_STABLE (SPELLING_VERSION_1_0)
#else
# define SPELLING_VERSION_PREV_STABLE (G_ENCODE_VERSION (SPELLING_MAJOR_VERSION - 1, 0))
#endif

/**
 * SPELLING_VERSION_MIN_REQUIRED:
 *
 * A macro that should be defined by the user prior to including
 * the spelling.h header.
 *
 * The definition should be one of the predefined SPELLING version
 * macros: %SPELLING_VERSION_1_0, ...
 *
 * This macro defines the lower bound for the Builder API to use.
 *
 * If a function has been deprecated in a newer version of Builder,
 * it is possible to use this symbol to avoid the compiler warnings
 * without disabling warning for every deprecated function.
 */
#ifndef SPELLING_VERSION_MIN_REQUIRED
# define SPELLING_VERSION_MIN_REQUIRED (SPELLING_VERSION_CUR_STABLE)
#endif

/**
 * SPELLING_VERSION_MAX_ALLOWED:
 *
 * A macro that should be defined by the user prior to including
 * the spelling.h header.

 * The definition should be one of the predefined Builder version
 * macros: %SPELLING_VERSION_1_0, %SPELLING_VERSION_1_2,...
 *
 * This macro defines the upper bound for the SPELLING API to use.
 *
 * If a function has been introduced in a newer version of Builder,
 * it is possible to use this symbol to get compiler warnings when
 * trying to use that function.
 */
#ifndef SPELLING_VERSION_MAX_ALLOWED
# if SPELLING_VERSION_MIN_REQUIRED > SPELLING_VERSION_PREV_STABLE
#  define SPELLING_VERSION_MAX_ALLOWED (SPELLING_VERSION_MIN_REQUIRED)
# else
#  define SPELLING_VERSION_MAX_ALLOWED (SPELLING_VERSION_CUR_STABLE)
# endif
#endif

#define SPELLING_AVAILABLE_IN_ALL _SPELLING_EXTERN

#if SPELLING_VERSION_MIN_REQUIRED >= SPELLING_VERSION_1_0
# define SPELLING_DEPRECATED_IN_1_0 SPELLING_DEPRECATED
# define SPELLING_DEPRECATED_IN_1_0_FOR(f) SPELLING_DEPRECATED_FOR(f)
#else
# define SPELLING_DEPRECATED_IN_1_0 _SPELLING_EXTERN
# define SPELLING_DEPRECATED_IN_1_0_FOR(f) _SPELLING_EXTERN
#endif
#if SPELLING_VERSION_MAX_ALLOWED < SPELLING_VERSION_1_0
# define SPELLING_AVAILABLE_IN_1_0 SPELLING_UNAVAILABLE(1, 0)
#else
# define SPELLING_AVAILABLE_IN_1_0 _SPELLING_EXTERN
#endif
