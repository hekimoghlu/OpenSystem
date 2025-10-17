/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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
#ifndef HFS_DBG_H_
#define HFS_DBG_H_

#include <sys/cdefs.h>

__BEGIN_DECLS

#include <stdbool.h>

// So that the analyzer acknowledges assertions...
#if defined(__clang_analyzer__) || DEBUG
#define panic_on_assert true
#else
extern bool panic_on_assert;
#endif

#if DEBUG
extern bool hfs_corruption_panics;
#else
#define hfs_corruption_panics false
#endif

__attribute__((noreturn))
void hfs_assert_fail(const char *file, unsigned line, const char *expr);

#define hfs_assert(expr)										\
	do {														\
		if (__builtin_expect(panic_on_assert, false)			\
			&& __builtin_expect(!(expr), false)) {				\
			hfs_assert_fail(__FILE__, __LINE__, #expr);			\
		}														\
	} while (0)

// On production, will printf rather than assert
#define hfs_warn(format, ...)									\
	do {														\
		if (__builtin_expect(panic_on_assert, false)) {			\
			panic(format, ## __VA_ARGS__);						\
			__builtin_unreachable();							\
		} else													\
			printf(format, ## __VA_ARGS__);						\
	} while (0)

// Quiet on production
#define hfs_debug(format, ...)									\
	do {														\
		if (__builtin_expect(panic_on_assert, false))			\
			printf(format, ## __VA_ARGS__);						\
	} while (0)

// Panic on debug unless boot-arg tells us not to
#define hfs_corruption_debug(format, ...)						\
	do {														\
		if (__builtin_expect(hfs_corruption_panics, false)) {	\
			panic(format, ## __VA_ARGS__);						\
			__builtin_unreachable();							\
		}														\
		else													\
			printf(format, ## __VA_ARGS__);						\
	} while (0)

__END_DECLS

#endif // HFS_DBG_H_
