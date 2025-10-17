/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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
/*
 * These are ABI implementations for when the raw iconv_* symbol
 * space was exposed via libc.so.7 in its early life.  This is
 * a transition aide, these wrappers will not normally ever be
 * executed except via __sym_compat() references.
 */
#include <sys/types.h>
#include <iconv.h>
#include "iconv-internal.h"

#ifndef __APPLE__
size_t
__iconv_compat(iconv_t a, char ** b, size_t * c, char ** d,
     size_t * e, __uint32_t f, size_t *g)
{
	return __bsd___iconv(a, b, c, d, e, f, g);
}

void
__iconv_free_list_compat(char ** a, size_t b)
{
	__bsd___iconv_free_list(a, b);
}

int
__iconv_get_list_compat(char ***a, size_t *b, __iconv_bool c)
{
	return __bsd___iconv_get_list(a, b, c);
}

size_t
iconv_compat(iconv_t a, char ** __restrict b,
      size_t * __restrict c, char ** __restrict d,
      size_t * __restrict e)
{
	return __bsd_iconv(a, b, c, d, e);
}

const char *
iconv_canonicalize_compat(const char *a)
{
	return __bsd_iconv_canonicalize(a);
}

int
iconv_close_compat(iconv_t a)
{
	return __bsd_iconv_close(a);
}

iconv_t
iconv_open_compat(const char *a, const char *b)
{
	return __bsd_iconv_open(a, b);
}

int
iconv_open_into_compat(const char *a, const char *b, iconv_allocation_t *c)
{
	return __bsd_iconv_open_into(a, b, c);
}

void
iconv_set_relocation_prefix_compat(const char *a, const char *b)
{
	return __bsd_iconv_set_relocation_prefix(a, b);
}

int
iconvctl_compat(iconv_t a, int b, void *c)
{
	return __bsd_iconvctl(a, b, c);
}

void
iconvlist_compat(int (*a) (unsigned int, const char * const *, void *), void *b)
{
	return __bsd_iconvlist(a, b);
}

int _iconv_version_compat = 0x0108;	/* Magic - not used */

__sym_compat(__iconv, __iconv_compat, FBSD_1.2);
__sym_compat(__iconv_free_list, __iconv_free_list_compat, FBSD_1.2);
__sym_compat(__iconv_get_list, __iconv_get_list_compat, FBSD_1.2);
__sym_compat(_iconv_version, _iconv_version_compat, FBSD_1.3);
__sym_compat(iconv, iconv_compat, FBSD_1.3);
__sym_compat(iconv_canonicalize, iconv_canonicalize_compat, FBSD_1.2);
__sym_compat(iconv_close, iconv_close_compat, FBSD_1.3);
__sym_compat(iconv_open, iconv_open_compat, FBSD_1.3);
__sym_compat(iconv_open_into, iconv_open_into_compat, FBSD_1.3);
__sym_compat(iconv_set_relocation_prefix, iconv_set_relocation_prefix_compat, FBSD_1.3);
__sym_compat(iconvctl, iconvctl_compat, FBSD_1.3);
__sym_compat(iconvlist, iconvlist_compat, FBSD_1.3);
#endif
