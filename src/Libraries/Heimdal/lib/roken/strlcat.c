/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 25, 2025.
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

#ifndef HAVE_STRLCAT

ROKEN_LIB_FUNCTION size_t ROKEN_LIB_CALL
strlcat (char *dst, const char *src, size_t dst_sz)
{
    size_t len;
#if defined(_MSC_VER) && _MSC_VER >= 1400
    len = strnlen_s(dst, dst_sz);
#elif defined(HAVE_STRNLEN)
    len = strnlen(dst, dst_sz);
#else
    len = strlen(dst);
#endif

    if (dst_sz <= len)
	/* the total size of dst is less than the string it contains;
           this could be considered bad input, but we might as well
           handle it */
	return len + strlen(src);

    return len + strlcpy (dst + len, src, dst_sz - len);
}

#endif
