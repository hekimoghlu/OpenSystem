/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 6, 2022.
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
#include <config.h>
#include "roken.h"

#ifndef HAVE_STRLCPY

#if defined(_MSC_VER) && _MSC_VER >= 1400

ROKEN_LIB_FUNCTION size_t ROKEN_LIB_CALL
strlcpy (char *dst, const char *src, size_t dst_cch)
{
    errno_t e;

    if (dst_cch > 0)
        e = strncpy_s(dst, dst_cch, src, _TRUNCATE);

    return strlen (src);
}

#else

ROKEN_LIB_FUNCTION size_t ROKEN_LIB_CALL
strlcpy (char *dst, const char *src, size_t dst_sz)
{
    size_t n;

    for (n = 0; n < dst_sz; n++) {
	if ((*dst++ = *src++) == '\0')
	    break;
    }

    if (n < dst_sz)
	return n;
    if (n > 0)
	*(dst - 1) = '\0';
    return n + strlen (src);
}

#endif

#endif
