/*
 * mks-version-macros.h
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include <glib.h>

#include "mks-version.h"

#ifndef _MKS_EXTERN
# define _MKS_EXTERN extern
#endif

#define MKS_VERSION_CUR_STABLE (G_ENCODE_VERSION (MKS_MAJOR_VERSION, 0))

#ifdef MKS_DISABLE_DEPRECATION_WARNINGS
# define MKS_DEPRECATED _MKS_EXTERN
# define MKS_DEPRECATED_FOR(f) _MKS_EXTERN
# define MKS_UNAVAILABLE(maj,min) _MKS_EXTERN
#else
# define MKS_DEPRECATED G_DEPRECATED _MKS_EXTERN
# define MKS_DEPRECATED_FOR(f) G_DEPRECATED_FOR (f) _MKS_EXTERN
# define MKS_UNAVAILABLE(maj,min) G_UNAVAILABLE (maj, min) _MKS_EXTERN
#endif

#define MKS_VERSION_1_0 (G_ENCODE_VERSION (1, 0))

#if MKS_MAJOR_VERSION == MKS_VERSION_1_0
# define MKS_VERSION_PREV_STABLE (MKS_VERSION_1_0)
#else
# define MKS_VERSION_PREV_STABLE (G_ENCODE_VERSION (MKS_MAJOR_VERSION - 1, 0))
#endif

/**
 * MKS_VERSION_MIN_REQUIRED:
 *
 * A macro that should be defined by the user prior to including
 * the mks.h header.
 *
 * The definition should be one of the predefined MKS version
 * macros: %MKS_VERSION_1_0, ...
 *
 * This macro defines the lower bound for the Builder API to use.
 *
 * If a function has been deprecated in a newer version of Builder,
 * it is possible to use this symbol to avoid the compiler warnings
 * without disabling warning for every deprecated function.
 */
#ifndef MKS_VERSION_MIN_REQUIRED
# define MKS_VERSION_MIN_REQUIRED (MKS_VERSION_CUR_STABLE)
#endif

/**
 * MKS_VERSION_MAX_ALLOWED:
 *
 * A macro that should be defined by the user prior to including
 * the mks.h header.

 * The definition should be one of the predefined Builder version
 * macros: %MKS_VERSION_1_0, %MKS_VERSION_44,...
 *
 * This macro defines the upper bound for the MKS API to use.
 *
 * If a function has been introduced in a newer version of Builder,
 * it is possible to use this symbol to get compiler warnings when
 * trying to use that function.
 */
#ifndef MKS_VERSION_MAX_ALLOWED
# if MKS_VERSION_MIN_REQUIRED > MKS_VERSION_PREV_STABLE
#  define MKS_VERSION_MAX_ALLOWED (MKS_VERSION_MIN_REQUIRED)
# else
#  define MKS_VERSION_MAX_ALLOWED (MKS_VERSION_CUR_STABLE)
# endif
#endif

#define MKS_AVAILABLE_IN_ALL _MKS_EXTERN

#if MKS_VERSION_MIN_REQUIRED >= MKS_VERSION_1_0
# define MKS_DEPRECATED_IN_1_0 MKS_DEPRECATED
# define MKS_DEPRECATED_IN_1_0_FOR(f) MKS_DEPRECATED_FOR(f)
#else
# define MKS_DEPRECATED_IN_1_0 _MKS_EXTERN
# define MKS_DEPRECATED_IN_1_0_FOR(f) _MKS_EXTERN
#endif
#if MKS_VERSION_MAX_ALLOWED < MKS_VERSION_1_0
# define MKS_AVAILABLE_IN_1_0 MKS_UNAVAILABLE(1, 0)
#else
# define MKS_AVAILABLE_IN_1_0 _MKS_EXTERN
#endif
