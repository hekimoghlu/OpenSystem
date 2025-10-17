/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#define __STDC_WANT_LIB_EXT1__ 1	/* for memset_s() */

#include <string.h>
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif /* HAVE_STRINGS_H */

#include "sudo_compat.h"

#ifndef HAVE_EXPLICIT_BZERO

# if defined(HAVE_EXPLICIT_MEMSET)
void
sudo_explicit_bzero(void *s, size_t n)
{
    explicit_memset(s, 0, n);
}
# elif defined(HAVE_MEMSET_EXPLICIT)
void
sudo_explicit_bzero(void *s, size_t n)
{
    memset_explicit(s, 0, n);
}
# elif defined(HAVE_MEMSET_S)
void
sudo_explicit_bzero(void *s, size_t n)
{
    (void)memset_s(s, n, 0, n);
}
# elif defined(HAVE_BZERO)
/* Jumping through a volatile function pointer should not be optimized away. */
void (* volatile sudo_explicit_bzero_impl)(void *, size_t) =
    (void (*)(void *, size_t))bzero;

void
sudo_explicit_bzero(void *s, size_t n)
{
    sudo_explicit_bzero_impl(s, n);
}
# else
void
sudo_explicit_bzero(void *v, size_t n)
{
    volatile unsigned char *s = v;

    /* Updating through a volatile pointer should not be optimized away. */
    while (n--)
	*s++ = '\0';
}
# endif /* HAVE_BZERO */

#endif /* HAVE_EXPLICIT_BZERO */
