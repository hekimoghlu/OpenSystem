/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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
#include "xlocale_private.h"

#include <ctype.h>
#include <string.h>
#include <rune.h>
#include <stdlib.h>
#include <wchar.h>
#include "runedepreciated.h"

__private_extern__ const char __rune_depreciated_msg[] = "\
%s and other functions prototyped in rune.h are depreciated in favor of\n\
the ISO C99 extended multibyte and wide character facilities and should not\n\
be used in new applications.\n\
";

__private_extern__ rune_t
__sgetrune(const char *string, size_t n, char const **result)
{
	wchar_t wc;
	size_t converted = mbrtowc(&wc, string, n, NULL);
	__darwin_rune_t invalid_rune =
	    __current_lc_ctype->_CurrentRuneLocale->__invalid_rune;

	switch (converted) {
	case (size_t)-2:	/* incomplete */
		if (result)
			*result = string;
		return invalid_rune;
	case (size_t)-1:	/* invalid */
		if (result)
			*result = string + 1;
		return invalid_rune;
	case (size_t)0:		/* null wide character */
	{
		int i;

		for (i = 1; i < n; i++)
			if (mbrtowc(&wc, string, n, NULL) == (size_t)0)
				break;
		if (result)
			*result = string + i;
		return (rune_t)0;
	}
	default:
		if (result)
			*result = string + converted;
		return (rune_t)wc;
	}
	/* NOTREACHED */
}

__private_extern__ int
__sputrune(rune_t rune, char *string, size_t n, char **result)
{
	char buf[MB_CUR_MAX];
	size_t converted = wcrtomb(buf, rune, NULL);

	if (converted == (size_t)-1) {
		if (result)
			*result = string;
	} else if (n >= converted) {
		if (string)
			bcopy(buf, string, converted);
		if (result)
			*result = string + converted;
	} else if (result)
		*result = NULL;
	return (converted == (size_t)-1 ? 0 : converted);
}

__private_extern__ rune_t
sgetrune(const char *string, size_t n, char const **result)
{
	static int warn_depreciated = 1;

	if (warn_depreciated) {
		warn_depreciated = 0;
		fprintf(stderr, __rune_depreciated_msg, "sgetrune");
	}
	return __sgetrune(string, n, result);
}

__private_extern__ int
sputrune(rune_t rune, char *string, size_t n, char **result)
{
	static int warn_depreciated = 1;

	if (warn_depreciated) {
		warn_depreciated = 0;
		fprintf(stderr, __rune_depreciated_msg, "sputrune");
	}
	return __sputrune(rune, string, n, result);
}
