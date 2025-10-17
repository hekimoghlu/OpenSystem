/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
#pragma prototyped

/*
 * localeconv() intercept
 */

#include "lclib.h"

#undef	localeconv

static char	null[] = "";

static struct lconv	debug_lconv =
{
	",",
	".",
	&null[0],
	&null[0],
	&null[0],
	&null[0],
	&null[0],
	&null[0],
	&null[0],
	&null[0],
	CHAR_MAX,
	CHAR_MAX,
	CHAR_MAX,
	CHAR_MAX,
	CHAR_MAX,
	CHAR_MAX,
	CHAR_MAX,
	CHAR_MAX,
};

static struct lconv	default_lconv =
{
	".",
	&null[0],
	&null[0],
	&null[0],
	&null[0],
	&null[0],
	&null[0],
	&null[0],
	&null[0],
	&null[0],
	CHAR_MAX,
	CHAR_MAX,
	CHAR_MAX,
	CHAR_MAX,
	CHAR_MAX,
	CHAR_MAX,
	CHAR_MAX,
	CHAR_MAX,
};

#if !_lib_localeconv

struct lconv*
localeconv(void)
{
	return &default_lconv;
}

#endif

/*
 * localeconv() intercept
 */

struct lconv*
_ast_localeconv(void)
{
	if ((locales[AST_LC_MONETARY]->flags | locales[AST_LC_NUMERIC]->flags) & LC_debug)
		return &debug_lconv;
	if ((locales[AST_LC_NUMERIC]->flags & (LC_default|LC_local)) == LC_local)
		return locales[AST_LC_NUMERIC]->territory == &lc_territories[0] ? &default_lconv : &debug_lconv;
	return localeconv();
}
