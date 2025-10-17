/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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
/* Written by Paul Eggert.  Modified by Chet Ramey for Bash. */

#if HAVE_CONFIG_H
#  include <config.h>
#endif

#if HAVE_INTTYPES_H
#  include <inttypes.h>
#endif

#if HAVE_STDLIB_H
#  include <stdlib.h>
#endif

#include <stdc.h>

/* Verify a requirement at compile-time (unlike assert, which is runtime).  */
#define verify(name, assertion) struct name { char a[(assertion) ? 1 : -1]; }

#ifndef HAVE_DECL_STRTOUL
"this configure-time declaration test was not run"
#endif
#if !HAVE_DECL_STRTOUL
extern unsigned long strtoul __P((const char *, char **, int));
#endif

#ifndef HAVE_DECL_STRTOULL
"this configure-time declaration test was not run"
#endif
#if !HAVE_DECL_STRTOULL && HAVE_UNSIGNED_LONG_LONG
extern unsigned long long strtoull __P((const char *, char **, int));
#endif

#ifdef strtoumax
#undef strtoumax
#endif

uintmax_t
strtoumax (ptr, endptr, base)
     const char *ptr;
     char **endptr;
     int base;
{
#if HAVE_UNSIGNED_LONG_LONG
  verify (size_is_that_of_unsigned_long_or_unsigned_long_long,
	  (sizeof (uintmax_t) == sizeof (unsigned long) ||
	   sizeof (uintmax_t) == sizeof (unsigned long long)));

  if (sizeof (uintmax_t) != sizeof (unsigned long))
    return (strtoull (ptr, endptr, base));
#else
  verify (size_is_that_of_unsigned_long, sizeof (uintmax_t) == sizeof (unsigned long));
#endif

  return (strtoul (ptr, endptr, base));
}

#ifdef TESTING
# include <stdio.h>
int
main ()
{
  char *p, *endptr;
  uintmax_t x;
#if HAVE_UNSIGNED_LONG_LONG
  unsigned long long y;
#endif
  unsigned long z;

  printf ("sizeof uintmax_t: %d\n", sizeof (uintmax_t));

#if HAVE_UNSIGNED_LONG_LONG
  printf ("sizeof unsigned long long: %d\n", sizeof (unsigned long long));
#endif
  printf ("sizeof unsigned long: %d\n", sizeof (unsigned long));

  x = strtoumax("42", &endptr, 10);
#if HAVE_LONG_LONG
  y = strtoull("42", &endptr, 10);
#else
  y = 0;
#endif
  z = strtoul("42", &endptr, 10);

  printf ("%llu %llu %lu\n", x, y, z);

  exit (0);
}
#endif
