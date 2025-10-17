/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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
#ifndef YASM_UTIL_H
#define YASM_UTIL_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#if defined(HAVE_GNU_C_LIBRARY) || defined(__MINGW32__) || defined(__DJGPP__)

/* Work around glibc's non-defining of certain things when using gcc -ansi */
# ifdef __STRICT_ANSI__
#  undef __STRICT_ANSI__
# endif

/* Work around glibc's string inlines (in bits/string2.h) if needed */
# ifdef NO_STRING_INLINES
#  define __NO_STRING_INLINES
# endif

#endif

#if !defined(lint) && !defined(NDEBUG)
# define NDEBUG
#endif

#include <stdio.h>
#include <stdarg.h>

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef HAVE_STRINGS_H
#include <strings.h>
#endif

#include <libyasm-stdint.h>
#include <libyasm/coretype.h>

#ifdef lint
# define _(String)      String
#else
# ifdef HAVE_LOCALE_H
#  include <locale.h>
# endif

# ifdef ENABLE_NLS
#  include <libintl.h>
#  define _(String)     gettext(String)
# else
#  define gettext(Msgid)                            (Msgid)
#  define dgettext(Domainname, Msgid)               (Msgid)
#  define dcgettext(Domainname, Msgid, Category)    (Msgid)
#  define textdomain(Domainname)                    while (0) /* nothing */
#  define bindtextdomain(Domainname, Dirname)       while (0) /* nothing */
#  define _(String)     (String)
# endif
#endif

#ifdef gettext_noop
# define N_(String)     gettext_noop(String)
#else
# define N_(String)     (String)
#endif

#ifdef HAVE_MERGESORT
#define yasm__mergesort(a, b, c, d)     mergesort(a, b, c, d)
#endif

#ifdef HAVE_STRSEP
#define yasm__strsep(a, b)              strsep(a, b)
#endif

#ifdef HAVE_STRCASECMP
# define yasm__strcasecmp(x, y)         strcasecmp(x, y)
# define yasm__strncasecmp(x, y, n)     strncasecmp(x, y, n)
#elif HAVE_STRICMP
# define yasm__strcasecmp(x, y)         stricmp(x, y)
# define yasm__strncasecmp(x, y, n)     strnicmp(x, y, n)
#elif HAVE__STRICMP
# define yasm__strcasecmp(x, y)         _stricmp(x, y)
# define yasm__strncasecmp(x, y, n)     _strnicmp(x, y, n)
#elif HAVE_STRCMPI
# define yasm__strcasecmp(x, y)         strcmpi(x, y)
# define yasm__strncasecmp(x, y, n)     strncmpi(x, y, n)
#else
# define USE_OUR_OWN_STRCASECMP
#endif

#include <libyasm/compat-queue.h>

#ifdef WITH_DMALLOC
# include <dmalloc.h>
# define yasm__xstrdup(str)             xstrdup(str)
# define yasm_xmalloc(size)             xmalloc(size)
# define yasm_xcalloc(count, size)      xcalloc(count, size)
# define yasm_xrealloc(ptr, size)       xrealloc(ptr, size)
# define yasm_xfree(ptr)                xfree(ptr)
#endif

/* Bit-counting: used primarily by HAMT but also in a few other places. */
#define BC_TWO(c)       (0x1ul << (c))
#define BC_MSK(c)       (((unsigned long)(-1)) / (BC_TWO(BC_TWO(c)) + 1ul))
#define BC_COUNT(x,c)   ((x) & BC_MSK(c)) + (((x) >> (BC_TWO(c))) & BC_MSK(c))
#define BitCount(d, s)          do {            \
        d = BC_COUNT(s, 0);                     \
        d = BC_COUNT(d, 1);                     \
        d = BC_COUNT(d, 2);                     \
        d = BC_COUNT(d, 3);                     \
        d = BC_COUNT(d, 4);                     \
    } while (0)

/** Determine if a value is exactly a power of 2.  Zero is treated as a power
 * of two.
 * \param x     value
 * \return Nonzero if x is a power of 2.
 */
#define is_exp2(x)  ((x & (x - 1)) == 0)

#ifndef NELEMS
/** Get the number of elements in an array.
 * \internal
 * \param array     array
 * \return Number of elements.
 */
#define NELEMS(array)   (sizeof(array) / sizeof(array[0]))
#endif

#endif
