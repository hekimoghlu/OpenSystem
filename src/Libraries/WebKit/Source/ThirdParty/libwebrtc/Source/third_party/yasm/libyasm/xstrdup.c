/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#include "util.h"

#include "coretype.h"

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)strdup.c    8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */


#ifdef WITH_DMALLOC
#undef yasm__xstrdup
#endif

char *
yasm__xstrdup(const char *str)
{
        size_t len;
        char *copy;

        len = strlen(str) + 1;
        copy = yasm_xmalloc(len);
        memcpy(copy, str, len);
        return (copy);
}

char *
yasm__xstrndup(const char *str, size_t max)
{
        size_t len = 0;
        char *copy;

        while (len < max && str[len] != '\0')
            len++;
        copy = yasm_xmalloc(len+1);
        memcpy(copy, str, len);
        copy[len] = '\0';
        return (copy);
}
