/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 8, 2024.
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
 * @file libgen.h
 * @brief POSIX basename() and dirname().
 *
 * This file contains the POSIX basename() and dirname(). See `<string.h>` for the GNU basename().
 */

#include <sys/cdefs.h>
#include <sys/types.h>

__BEGIN_DECLS

/**
 * [basename(3)](https://man7.org/linux/man-pages/man3/basename.3.html)
 * returns the final component of the given path.
 *
 * See `<string.h>` for the GNU basename(). Including `<libgen.h>`,
 * either before or after including <string.h>, will override the GNU variant.
 *
 * Note that Android's cv-qualifiers differ from POSIX; Android's implementation doesn't
 * modify its input and uses thread-local storage for the result if necessary.
 */
char* _Nullable __posix_basename(const char* _Nullable __path) __RENAME(basename);

/**
 * This macro ensures that callers get the POSIX basename() if they include this header,
 * no matter what order `<libgen.h>` and `<string.h>` are included in.
 */
#define basename __posix_basename

/**
 * [dirname(3)](https://man7.org/linux/man-pages/man3/dirname.3.html)
 * returns all but the final component of the given path.
 *
 * Note that Android's cv-qualifiers differ from POSIX; Android's implementation doesn't
 * modify its input and uses thread-local storage for the result if necessary.
 */
char* _Nullable dirname(const char* _Nullable __path);

#if !defined(__LP64__)
/** Deprecated. Use dirname() instead. */
int dirname_r(const char* _Nullable __path, char* _Nullable __buf, size_t __n);
/** Deprecated. Use basename() instead. */
int basename_r(const char* _Nullable __path, char* _Nullable __buf, size_t __n);
#endif

__END_DECLS
