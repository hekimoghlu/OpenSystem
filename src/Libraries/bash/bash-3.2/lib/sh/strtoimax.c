/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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

#ifndef HAVE_DECL_STRTOL
"this configure-time declaration test was not run"
#endif
#if !HAVE_DECL_STRTOL
extern long strtol __P((const char *, char **, int));
#endif

#ifndef HAVE_DECL_STRTOLL
"this configure-time declaration test was not run"
#endif
#if !HAVE_DECL_STRTOLL && HAVE_LONG_LONG
extern long long strtoll __P((const char *, char **, int));
#endif

#ifdef strtoimax
#undef strtoimax
#endif

intmax_t
strtoimax (ptr, endptr, base)
     const char *ptr;
     char **endptr;
     int base;
{
#if HAVE_LONG_LONG
  verify(size_is_that_of_long_or_long_long,
	 (sizeof (intmax_t) == sizeof (long) ||
	  sizeof (intmax_t) == sizeof (long long)));

  if (sizeof (intmax_t) != sizeof (long))
    return (strtoll (ptr, endptr, base));
#else
  verify (size_is_that_of_long, sizeof (intmax_t) == sizeof (long));
#endif

  return (strtol (ptr, endptr, base));
}

#ifdef TESTING
# include <stdio.h>
int
main ()
{
  char *p, *endptr;
  intmax_t x;
#if HAVE_LONG_LONG
  long long y;
#endif
  long z;
  
  printf ("sizeof intmax_t: %d\n", sizeof (intmax_t));

#if HAVE_LONG_LONG
  printf ("sizeof long long: %d\n", sizeof (long long));
#endif
  printf ("sizeof long: %d\n", sizeof (long));

  x = strtoimax("42", &endptr, 10);
#if HAVE_LONG_LONG
  y = strtoll("42", &endptr, 10);
#else
  y = -1;
#endif
  z = strtol("42", &endptr, 10);

  printf ("%lld %lld %ld\n", x, y, z);

  exit (0);
}
#endif
