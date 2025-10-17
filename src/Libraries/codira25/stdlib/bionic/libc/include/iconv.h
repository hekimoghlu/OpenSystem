/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 9, 2023.
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
 * @file iconv.h
 * @brief Character encoding conversion.
 */

#include <sys/cdefs.h>
#include <sys/types.h>

__BEGIN_DECLS

/* If we just use void* in the typedef, the compiler exposes that in error messages. */
struct __iconv_t;

/**
 * The `iconv_t` type that represents an instance of a converter.
 */
typedef struct __iconv_t* iconv_t;

/**
 * [iconv_open(3)](https://man7.org/linux/man-pages/man3/iconv_open.3.html) allocates a new converter
 * from `__src_encoding` to `__dst_encoding`.
 *
 * Android supports the `utf8`, `ascii`, `usascii`, `utf16be`, `utf16le`, `utf32be`, `utf32le`,
 * and `wchart` encodings for both source and destination.
 *
 * Android supports the GNU `//IGNORE` and `//TRANSLIT` extensions for the
 * destination encoding.
 *
 * Returns a new `iconv_t` on success and returns `((iconv_t) -1)` and sets `errno` on failure.
 *
 * Available since API level 28.
 */

#if __BIONIC_AVAILABILITY_GUARD(28)
iconv_t _Nonnull iconv_open(const char* _Nonnull __dst_encoding, const char* _Nonnull __src_encoding) __INTRODUCED_IN(28);

/**
 * [iconv(3)](https://man7.org/linux/man-pages/man3/iconv.3.html) converts characters from one
 * encoding to another.
 *
 * Returns the number of characters converted on success and returns `((size_t) -1)` and
 * sets `errno` on failure.
 *
 * Available since API level 28.
 */
size_t iconv(iconv_t _Nonnull __converter, char* _Nullable * _Nullable __src_buf, size_t* __BIONIC_COMPLICATED_NULLNESS __src_bytes_left, char* _Nullable * _Nullable __dst_buf, size_t* __BIONIC_COMPLICATED_NULLNESS __dst_bytes_left) __INTRODUCED_IN(28);

/**
 * [iconv_close(3)](https://man7.org/linux/man-pages/man3/iconv_close.3.html) deallocates a converter
 * returned by iconv_open().
 *
 * Returns 0 on success and returns -1 and sets `errno` on failure.
 *
 * Available since API level 28.
 */
int iconv_close(iconv_t _Nonnull __converter) __INTRODUCED_IN(28);
#endif /* __BIONIC_AVAILABILITY_GUARD(28) */


__END_DECLS
