/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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
 * @file fnmatch.h
 * @brief Filename matching.
 */

#include <sys/cdefs.h>

__BEGIN_DECLS

/** Returned by fnmatch() if matching failed. */
#define FNM_NOMATCH 1

/** Returned by fnmatch() if the function is not supported. This is never returned on Android. */
#define FNM_NOSYS 2

/** fnmatch() flag to disable backslash escaping. */
#define FNM_NOESCAPE     0x01
/** fnmatch() flag to ensure that slashes must be matched by slashes. */
#define FNM_PATHNAME     0x02
/** fnmatch() flag to ensure that periods must be matched by periods. */
#define FNM_PERIOD       0x04
/** fnmatch() flag to ignore /... after a match. */
#define FNM_LEADING_DIR  0x08
/** fnmatch() flag for a case-insensitive search. */
#define FNM_CASEFOLD     0x10

/** Synonym for `FNM_CASEFOLD`: case-insensitive search. */
#define FNM_IGNORECASE   FNM_CASEFOLD
/** Synonym for `FNM_PATHNAME`: slashes must be matched by slashes. */
#define FNM_FILE_NAME    FNM_PATHNAME

/**
 * [fnmatch(3)](https://man7.org/linux/man-pages/man3/fnmatch.3.html) matches `__string` against
 * the shell wildcard `__pattern`.
 *
 * Returns 0 on success, and returns `FNM_NOMATCH` on failure.
 */
int fnmatch(const char* _Nonnull __pattern, const char* _Nonnull __string, int __flags);

__END_DECLS
