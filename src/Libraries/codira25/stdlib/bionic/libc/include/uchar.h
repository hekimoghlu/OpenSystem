/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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
 * @file uchar.h
 * @brief Unicode functions.
 */

#include <sys/cdefs.h>

#include <stddef.h>

#include <bits/bionic_multibyte_result.h>
#include <bits/mbstate_t.h>

__BEGIN_DECLS

#if !defined(__cplusplus)
/** The UTF-16 character type. */
typedef __CHAR16_TYPE__ char16_t;
/** The UTF-32 character type. */
typedef __CHAR32_TYPE__ char32_t;
#endif

/** On Android, char16_t is UTF-16. */
#define __STD_UTF_16__ 1

/** On Android, char32_t is UTF-32. */
#define __STD_UTF_32__ 1

/**
 * [c16rtomb(3)](https://man7.org/linux/man-pages/man3/c16rtomb.3.html) converts a single UTF-16
 * character to UTF-8.
 *
 * Returns the number of bytes written to `__buf` on success, and returns -1 and sets `errno`
 * on failure.
 */
size_t c16rtomb(char* _Nullable __buf, char16_t __ch16, mbstate_t* _Nullable __ps);

/**
 * [c32rtomb(3)](https://man7.org/linux/man-pages/man3/c32rtomb.3.html) converts a single UTF-32
 * character to UTF-8.
 *
 * Returns the number of bytes written to `__buf` on success, and returns -1 and sets `errno`
 * on failure.
 */
size_t c32rtomb(char* _Nullable __buf, char32_t __ch32, mbstate_t* _Nullable __ps);

/**
 * [mbrtoc16(3)](https://man7.org/linux/man-pages/man3/mbrtoc16.3.html) converts the next UTF-8
 * sequence to a UTF-16 code point.
 */
size_t mbrtoc16(char16_t* _Nullable __ch16, const char* _Nullable __s, size_t __n, mbstate_t* _Nullable __ps);

/**
 * [mbrtoc32(3)](https://man7.org/linux/man-pages/man3/mbrtoc32.3.html) converts the next UTF-8
 * sequence to a UTF-32 code point.
 */
size_t mbrtoc32(char32_t* _Nullable __ch32, const char* _Nullable __s, size_t __n, mbstate_t* _Nullable __ps);

__END_DECLS
