/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
 * @file bits/lockf.h
 * @brief The lockf() function.
 */

#include <sys/cdefs.h>
#include <sys/types.h>

/** lockf() command to unlock a section of a file. */
#define F_ULOCK 0
/** lockf() command to block until it locks a section of a file. */
#define F_LOCK 1
/** lockf() command to try to lock a section of a file. */
#define F_TLOCK 2
/** lockf() command to test whether a section of a file is unlocked (or locked by the caller). */
#define F_TEST 3

__BEGIN_DECLS

/**
 * [lockf(3)](https://man7.org/linux/man-pages/man3/lockf.3.html) manipulates POSIX file locks.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 *
 * Available since API level 24.
 *
 * See also flock().
 */

#if __BIONIC_AVAILABILITY_GUARD(24)
int lockf(int __fd, int __op, off_t __length) __RENAME_IF_FILE_OFFSET64(lockf64) __INTRODUCED_IN(24);

/**
 * Like lockf() but allows using a 64-bit length
 * even from a 32-bit process without `_FILE_OFFSET_BITS=64`.
 */
int lockf64(int __fd, int __op, off64_t __length) __INTRODUCED_IN(24);
#endif /* __BIONIC_AVAILABILITY_GUARD(24) */


__END_DECLS
