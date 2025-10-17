/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 28, 2023.
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

#ifndef USE_OUR_OWN_STRCASECMP
#undef yasm__strcasecmp
#undef yasm__strncasecmp
#endif

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)strcasecmp.c        8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */

#include <ctype.h>

int
yasm__strcasecmp(const char *s1, const char *s2)
{
#ifdef HAVE_STRCASECMP
    return strcasecmp(s1, s2);
#elif HAVE_STRICMP
    return stricmp(s1, s2);
#elif HAVE__STRICMP
    return _stricmp(s1, s2);
#elif HAVE_STRCMPI
    return strcmpi(s1, s2);
#else
        const unsigned char
                        *us1 = (const unsigned char *)s1,
                        *us2 = (const unsigned char *)s2;

        while (tolower(*us1) == tolower(*us2++))
                if (*us1++ == '\0')
                        return (0);
        return (tolower(*us1) - tolower(*--us2));
#endif
}

int
yasm__strncasecmp(const char *s1, const char *s2, size_t n)
{
#ifdef HAVE_STRCASECMP
    return strncasecmp(s1, s2, n);
#elif HAVE_STRICMP
    return strnicmp(s1, s2, n);
#elif HAVE__STRNICMP
    return _strnicmp(s1, s2, n);
#elif HAVE_STRCMPI
    return strncmpi(s1, s2, n);
#else
        const unsigned char
                        *us1 = (const unsigned char *)s1,
                        *us2 = (const unsigned char *)s2;

        if (n != 0) {
                do {
                        if (tolower(*us1) != tolower(*us2++))
                                return (tolower(*us1) - tolower(*--us2));
                        if (*us1++ == '\0')
                                break;
                } while (--n != 0);
        }
        return (0);
#endif
}
