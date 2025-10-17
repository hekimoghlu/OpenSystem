/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#pragma once

/**
 * @file sys/param.h
 * @brief Various macros.
 */

#include <sys/cdefs.h>

#include <endian.h>
#include <limits.h>
#include <linux/param.h>

/** The unit of `st_blocks` in `struct stat`. */
#define DEV_BSIZE 512

/** A historical name for PATH_MAX. Use PATH_MAX in new code. */
#define MAXPATHLEN PATH_MAX

/** A historical name for NGROUPS_MAX. Use NGROUPS_MAX in new code. */
#define NGROUPS NGROUPS_MAX

#define MAXSYMLINKS 8

#ifndef howmany
#define howmany(x, y)   (((x)+((y)-1))/(y))
#endif
#define roundup(x, y)   ((((x)+((y)-1))/(y))*(y))

/**
 * Returns true if the binary representation of the argument is all zeros
 * or has exactly one bit set. Contrary to the macro name, this macro
 * DOES NOT determine if the provided value is a power of 2. In particular,
 * this function falsely returns true for powerof2(0) and some negative
 * numbers.
 */
#define powerof2(x)                                               \
  ({                                                              \
    __typeof__(x) _x = (x);                                       \
    __typeof__(x) _x2;                                            \
    __builtin_add_overflow(_x, -1, &_x2) ? 1 : ((_x2 & _x) == 0); \
  })

/** Returns the lesser of its two arguments. */
#define MIN(a,b) (((a)<(b))?(a):(b))
/** Returns the greater of its two arguments. */
#define MAX(a,b) (((a)>(b))?(a):(b))
