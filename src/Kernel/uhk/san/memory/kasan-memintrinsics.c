/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 28, 2024.
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
#include <string.h>
#include <mach/boolean.h>

#include <mach/boolean.h>
#include <machine/limits.h>
#include <kern/debug.h>

#include "kasan_internal.h"
#include "memintrinsics.h"

void
__asan_bcopy(const void *src, void *dst, size_t sz)
{
	kasan_check_range(src, sz, TYPE_MEMR);
	kasan_check_range(dst, sz, TYPE_MEMW);
	__nosan_bcopy(src, dst, sz);
}

void *
__asan_memmove(void *src, const void *dst, size_t sz)
{
	kasan_check_range(src, sz, TYPE_MEMR);
	kasan_check_range(dst, sz, TYPE_MEMW);
	return __nosan_memmove(src, dst, sz);
}

void *
__asan_memcpy(void *dst, const void *src, size_t sz)
{
	kasan_check_range(src, sz, TYPE_MEMR);
	kasan_check_range(dst, sz, TYPE_MEMW);
	return __nosan_memcpy(dst, src, sz);
}

void *
__asan_memset(void *dst, int c, size_t sz)
{
	kasan_check_range(dst, sz, TYPE_MEMW);
	return __nosan_memset(dst, c, sz);
}

void
__asan_bzero(void *dst, size_t sz)
{
	kasan_check_range(dst, sz, TYPE_MEMW);
	__nosan_bzero(dst, sz);
}

int
__asan_bcmp(const void *a, const void *b, size_t len)
{
	kasan_check_range(a, len, TYPE_MEMR);
	kasan_check_range(b, len, TYPE_MEMR);
	return __nosan_bcmp(a, b, len);
}

int
__asan_memcmp(const void *a, const void *b, size_t n)
{
	kasan_check_range(a, n, TYPE_MEMR);
	kasan_check_range(b, n, TYPE_MEMR);
	return __nosan_memcmp(a, b, n);
}

size_t
__asan_strlcpy(char *dst, const char *src, size_t sz)
{
	kasan_check_range(dst, sz, TYPE_STRW);
	return __nosan_strlcpy(dst, src, sz);
}

size_t
__asan_strlcat(char *dst, const char *src, size_t sz)
{
	kasan_check_range(dst, sz, TYPE_STRW);
	return __nosan_strlcat(dst, src, sz);
}

char *
__asan_strncpy(char *dst, const char *src, size_t sz)
{
	kasan_check_range(dst, sz, TYPE_STRW);
	return __nosan_strncpy(dst, src, sz);
}

char *
__asan_strncat(char *dst, const char *src, size_t sz)
{
	kasan_check_range(dst, strlen(dst) + sz + 1, TYPE_STRW);
	return __nosan_strncat(dst, src, sz);
}

size_t
__asan_strnlen(const char *src, size_t sz)
{
	kasan_check_range(src, sz, TYPE_STRR);
	return __nosan_strnlen(src, sz);
}

size_t
__asan_strlen(const char *src)
{
	size_t sz = __nosan_strlen(src);
	kasan_check_range(src, sz + 1, TYPE_STRR);
	return sz;
}
