/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
#include "includes.h"

/*
 * Don't let systems with broken printf(3) avoid our replacements
 * via asprintf(3)/vasprintf(3) calling libc internally.
 */
#if defined(BROKEN_SNPRINTF)
# undef HAVE_VASPRINTF
# undef HAVE_ASPRINTF
#endif

#ifndef HAVE_VASPRINTF

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#define INIT_SZ	128

int
vasprintf(char **str, const char *fmt, va_list ap)
{
	int ret = -1;
	va_list ap2;
	char *string, *newstr;
	size_t len;

	VA_COPY(ap2, ap);
	if ((string = malloc(INIT_SZ)) == NULL)
		goto fail;

	ret = vsnprintf(string, INIT_SZ, fmt, ap2);
	if (ret >= 0 && ret < INIT_SZ) { /* succeeded with initial alloc */
		*str = string;
	} else if (ret == INT_MAX || ret < 0) { /* Bad length */
		free(string);
		goto fail;
	} else {	/* bigger than initial, realloc allowing for nul */
		len = (size_t)ret + 1;
		if ((newstr = realloc(string, len)) == NULL) {
			free(string);
			goto fail;
		} else {
			va_end(ap2);
			VA_COPY(ap2, ap);
			ret = vsnprintf(newstr, len, fmt, ap2);
			if (ret >= 0 && (size_t)ret < len) {
				*str = newstr;
			} else { /* failed with realloc'ed string, give up */
				free(newstr);
				goto fail;
			}
		}
	}
	va_end(ap2);
	return (ret);

fail:
	*str = NULL;
	errno = ENOMEM;
	va_end(ap2);
	return (-1);
}
#endif

#ifndef HAVE_ASPRINTF
int asprintf(char **str, const char *fmt, ...)
{
	va_list ap;
	int ret;

	*str = NULL;
	va_start(ap, fmt);
	ret = vasprintf(str, fmt, ap);
	va_end(ap);

	return ret;
}
#endif
