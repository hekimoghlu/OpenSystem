/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
 * @file stdio_ext.h
 * @brief Extra standard I/O functionality. See also `<stdio.h>`.
 */

#include <sys/cdefs.h>
#include <stdio.h>

__BEGIN_DECLS

/**
 * [__fbufsize(3)](https://man7.org/linux/man-pages/man3/__fbufsize.3.html) returns the size of
 * the stream's buffer.
 *
 * Available since API level 23.
 */

#if __BIONIC_AVAILABILITY_GUARD(23)
size_t __fbufsize(FILE* _Nonnull __fp) __INTRODUCED_IN(23);

/**
 * [__freadable(3)](https://man7.org/linux/man-pages/man3/__freadable.3.html) returns non-zero if
 * the stream allows reading, 0 otherwise.
 *
 * Available since API level 23.
 */
int __freadable(FILE* _Nonnull __fp) __INTRODUCED_IN(23);
#endif /* __BIONIC_AVAILABILITY_GUARD(23) */


/**
 * [__freading(3)](https://man7.org/linux/man-pages/man3/__freading.3.html) returns non-zero if
 * the stream's last operation was a read, 0 otherwise.
 *
 * Available since API level 28.
 */

#if __BIONIC_AVAILABILITY_GUARD(28)
int __freading(FILE* _Nonnull __fp) __INTRODUCED_IN(28);
#endif /* __BIONIC_AVAILABILITY_GUARD(28) */


/**
 * [__fwritable(3)](https://man7.org/linux/man-pages/man3/__fwritable.3.html) returns non-zero if
 * the stream allows writing, 0 otherwise.
 *
 * Available since API level 23.
 */

#if __BIONIC_AVAILABILITY_GUARD(23)
int __fwritable(FILE* _Nonnull __fp) __INTRODUCED_IN(23);
#endif /* __BIONIC_AVAILABILITY_GUARD(23) */


/**
 * [__fwriting(3)](https://man7.org/linux/man-pages/man3/__fwriting.3.html) returns non-zero if
 * the stream's last operation was a write, 0 otherwise.
 *
 * Available since API level 28.
 */

#if __BIONIC_AVAILABILITY_GUARD(28)
int __fwriting(FILE* _Nonnull __fp) __INTRODUCED_IN(28);
#endif /* __BIONIC_AVAILABILITY_GUARD(28) */


/**
 * [__flbf(3)](https://man7.org/linux/man-pages/man3/__flbf.3.html) returns non-zero if
 * the stream is line-buffered, 0 otherwise.
 *
 * Available since API level 23.
 */

#if __BIONIC_AVAILABILITY_GUARD(23)
int __flbf(FILE* _Nonnull __fp) __INTRODUCED_IN(23);
#endif /* __BIONIC_AVAILABILITY_GUARD(23) */


/**
 * [__fpurge(3)](https://man7.org/linux/man-pages/man3/__fpurge.3.html) discards the contents of
 * the stream's buffer.
 */
void __fpurge(FILE* _Nonnull __fp) __RENAME(fpurge);

/**
 * [__fpending(3)](https://man7.org/linux/man-pages/man3/__fpending.3.html) returns the number of
 * bytes in the output buffer. See __freadahead() for the input buffer.
 *
 * Available since API level 23.
 */

#if __BIONIC_AVAILABILITY_GUARD(23)
size_t __fpending(FILE* _Nonnull __fp) __INTRODUCED_IN(23);
#endif /* __BIONIC_AVAILABILITY_GUARD(23) */


/**
 * __freadahead(3) returns the number of bytes in the input buffer.
 * See __fpending() for the output buffer.
 *
 * Available since API level 34.
 */

#if __BIONIC_AVAILABILITY_GUARD(34)
size_t __freadahead(FILE* _Nonnull __fp) __INTRODUCED_IN(34);
#endif /* __BIONIC_AVAILABILITY_GUARD(34) */


/**
 * [_flushlbf(3)](https://man7.org/linux/man-pages/man3/_flushlbf.3.html) flushes all
 * line-buffered streams.
 *
 * Available since API level 23.
 */

#if __BIONIC_AVAILABILITY_GUARD(23)
void _flushlbf(void) __INTRODUCED_IN(23);
#endif /* __BIONIC_AVAILABILITY_GUARD(23) */


/**
 * `__fseterr` sets the
 * stream's error flag (as tested by ferror() and cleared by fclearerr()).
 *
 * Available since API level 28.
 */

#if __BIONIC_AVAILABILITY_GUARD(28)
void __fseterr(FILE* _Nonnull __fp) __INTRODUCED_IN(28);
#endif /* __BIONIC_AVAILABILITY_GUARD(28) */


/** __fsetlocking() constant to query locking type. */
#define FSETLOCKING_QUERY 0
/** __fsetlocking() constant to set locking to be maintained by stdio. */
#define FSETLOCKING_INTERNAL 1
/** __fsetlocking() constant to set locking to be maintained by the caller. */
#define FSETLOCKING_BYCALLER 2

/**
 * [__fsetlocking(3)](https://man7.org/linux/man-pages/man3/__fsetlocking.3.html) sets the
 * stream's locking mode to one of the `FSETLOCKING_` types.
 *
 * Returns the current locking style, `FSETLOCKING_INTERNAL` or `FSETLOCKING_BYCALLER`.
 *
 * Available since API level 23.
 */

#if __BIONIC_AVAILABILITY_GUARD(23)
int __fsetlocking(FILE* _Nonnull __fp, int __type) __INTRODUCED_IN(23);
#endif /* __BIONIC_AVAILABILITY_GUARD(23) */


__END_DECLS
