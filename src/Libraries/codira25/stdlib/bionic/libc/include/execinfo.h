/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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

#include <sys/cdefs.h>

/**
 * @file execinfo.h
 * @brief Functions to do in process backtracing.
 */

__BEGIN_DECLS

/**
 * [backtrace(3)](https://man7.org/linux/man-pages/man3/backtrace.3.html)
 * Saves a backtrace for the current call in the array pointed to by buffer.
 * "size" indicates the maximum number of void* pointers that can be set.
 *
 * Returns the number of addresses stored in "buffer", which is not greater
 * than "size". If the return value is equal to "size" then the number of
 * addresses may have been truncated.
 *
 * Available since API level 33.
 */

#if __BIONIC_AVAILABILITY_GUARD(33)
int backtrace(void* _Nonnull * _Nonnull buffer, int size) __INTRODUCED_IN(33);

/**
 * [backtrace_symbols(3)](https://man7.org/linux/man-pages/man3/backtrace_symbols.3.html)
 * Given an array of void* pointers, translate the addresses into an array
 * of strings that represent the backtrace.
 *
 * Returns a pointer to allocated memory, on error NULL is returned. It is
 * the responsibility of the caller to free the returned memory.
 *
 * Available since API level 33.
 */
char* _Nullable * _Nullable backtrace_symbols(void* _Nonnull const* _Nonnull buffer, int size) __INTRODUCED_IN(33);

/**
 * [backtrace_symbols_fd(3)](https://man7.org/linux/man-pages/man3/backtrace_symbols_fd.3.html)
 * Given an array of void* pointers, translate the addresses into an array
 * of strings that represent the backtrace and write to the file represented
 * by "fd". The file is written such that one line equals one void* address.
 *
 * Available since API level 33.
 */
void backtrace_symbols_fd(void* _Nonnull const* _Nonnull buffer, int size, int fd) __INTRODUCED_IN(33);
#endif /* __BIONIC_AVAILABILITY_GUARD(33) */


__END_DECLS
