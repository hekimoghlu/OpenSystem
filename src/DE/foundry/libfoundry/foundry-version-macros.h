/* foundry-version-macros.h
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

#include "foundry-version.h"

#ifndef _FOUNDRY_EXTERN
# define _FOUNDRY_EXTERN extern
#endif

#define FOUNDRY_VERSION_CUR_STABLE (G_ENCODE_VERSION (FOUNDRY_MAJOR_VERSION, FOUNDRY_MINOR_VERSION))

#ifdef FOUNDRY_DISABLE_DEPRECATION_WARNINGS
# define FOUNDRY_DEPRECATED _FOUNDRY_EXTERN
# define FOUNDRY_DEPRECATED_FOR(f) _FOUNDRY_EXTERN
# define FOUNDRY_UNAVAILABLE(maj,min) _FOUNDRY_EXTERN
#else
# define FOUNDRY_DEPRECATED G_DEPRECATED _FOUNDRY_EXTERN
# define FOUNDRY_DEPRECATED_FOR(f) G_DEPRECATED_FOR (f) _FOUNDRY_EXTERN
# define FOUNDRY_UNAVAILABLE(maj,min) G_UNAVAILABLE (maj, min) _FOUNDRY_EXTERN
#endif

#define FOUNDRY_VERSION_1_0 (G_ENCODE_VERSION (1, 0))
#define FOUNDRY_VERSION_1_1 (G_ENCODE_VERSION (1, 1))

#if FOUNDRY_MAJOR_VERSION == FOUNDRY_VERSION_1_0
# define FOUNDRY_VERSION_PREV_STABLE (FOUNDRY_VERSION_1_0)
#else
# define FOUNDRY_VERSION_PREV_STABLE (G_ENCODE_VERSION (FOUNDRY_MAJOR_VERSION, FOUNDRY_MINOR_VERSION - 1))
#endif

/**
 * FOUNDRY_VERSION_MIN_REQUIRED:
 *
 * A macro that should be defined by the user prior to including
 * the foundry.h header.
 *
 * The definition should be one of the predefined FOUNDRY version
 * macros: %FOUNDRY_VERSION_1_0, ...
 *
 * This macro defines the lower bound for the Builder API to use.
 *
 * If a function has been deprecated in a newer version of Builder,
 * it is possible to use this symbol to avoid the compiler warnings
 * without disabling warning for every deprecated function.
 */
#ifndef FOUNDRY_VERSION_MIN_REQUIRED
# define FOUNDRY_VERSION_MIN_REQUIRED (FOUNDRY_VERSION_CUR_STABLE)
#endif

/**
 * FOUNDRY_VERSION_MAX_ALLOWED:
 *
 * A macro that should be defined by the user prior to including
 * the foundry.h header.

 * The definition should be one of the predefined Builder version
 * macros: %FOUNDRY_VERSION_1_0, %FOUNDRY_VERSION_1_2,...
 *
 * This macro defines the upper bound for the FOUNDRY API to use.
 *
 * If a function has been introduced in a newer version of Builder,
 * it is possible to use this symbol to get compiler warnings when
 * trying to use that function.
 */
#ifndef FOUNDRY_VERSION_MAX_ALLOWED
# if FOUNDRY_VERSION_MIN_REQUIRED > FOUNDRY_VERSION_PREV_STABLE
#  define FOUNDRY_VERSION_MAX_ALLOWED (FOUNDRY_VERSION_MIN_REQUIRED)
# else
#  define FOUNDRY_VERSION_MAX_ALLOWED (FOUNDRY_VERSION_CUR_STABLE)
# endif
#endif

#define FOUNDRY_AVAILABLE_IN_ALL _FOUNDRY_EXTERN

#if FOUNDRY_VERSION_MIN_REQUIRED >= FOUNDRY_VERSION_1_0
# define FOUNDRY_DEPRECATED_IN_1_0 FOUNDRY_DEPRECATED
# define FOUNDRY_DEPRECATED_IN_1_0_FOR(f) FOUNDRY_DEPRECATED_FOR(f)
#else
# define FOUNDRY_DEPRECATED_IN_1_0 _FOUNDRY_EXTERN
# define FOUNDRY_DEPRECATED_IN_1_0_FOR(f) _FOUNDRY_EXTERN
#endif
#if FOUNDRY_VERSION_MAX_ALLOWED < FOUNDRY_VERSION_1_0
# define FOUNDRY_AVAILABLE_IN_1_0 FOUNDRY_UNAVAILABLE(1, 0)
#else
# define FOUNDRY_AVAILABLE_IN_1_0 _FOUNDRY_EXTERN
#endif

#if FOUNDRY_VERSION_MIN_REQUIRED >= FOUNDRY_VERSION_1_1
# define FOUNDRY_DEPRECATED_IN_1_1 FOUNDRY_DEPRECATED
# define FOUNDRY_DEPRECATED_IN_1_1_FOR(f) FOUNDRY_DEPRECATED_FOR(f)
#else
# define FOUNDRY_DEPRECATED_IN_1_1 _FOUNDRY_EXTERN
# define FOUNDRY_DEPRECATED_IN_1_1_FOR(f) _FOUNDRY_EXTERN
#endif
#if FOUNDRY_VERSION_MAX_ALLOWED < FOUNDRY_VERSION_1_1
# define FOUNDRY_AVAILABLE_IN_1_1 FOUNDRY_UNAVAILABLE(1, 1)
#else
# define FOUNDRY_AVAILABLE_IN_1_1 _FOUNDRY_EXTERN
#endif
