/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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
#include "uucp.h"

#include <ctype.h>
#include <errno.h>

#if HAVE_LIMITS_H
#include <limits.h>
#else
#define ULONG_MAX 4294967295
#define LONG_MIN (- LONG_MAX - 1)
#define LONG_MAX 2147483647
#endif

#ifndef	UNSIGNED
#define	UNSIGNED	0
#endif

/* Convert NPTR to an `unsigned long int' or `long int' in base BASE.
   If BASE is 0 the base is determined by the presence of a leading
   zero, indicating octal or a leading "0x" or "0X", indicating hexadecimal.
   If BASE is < 2 or > 36, it is reset to 10.
   If ENDPTR is not NULL, a pointer to the character after the last
   one converted is stored in *ENDPTR.  */
#if	UNSIGNED
unsigned long int
#define	strtol	strtoul
#else
long int
#endif
strtol (nptr, endptr, base)
     const char *nptr;
     char **endptr;
     int base;
{
  int negative;
  register unsigned long int cutoff;
  register unsigned int cutlim;
  register unsigned long int i;
  register const char *s;
  register unsigned int c;
  const char *save;
  int overflow;

  if (base < 0 || base == 1 || base > 36)
    base = 10;

  s = nptr;

  /* Skip white space.  */
  while (isspace(BUCHAR (*s)))
    ++s;
  if (*s == '\0')
    goto noconv;

  /* Check for a sign.  */
  if (*s == '-')
    {
      negative = 1;
      ++s;
    }
  else if (*s == '+')
    {
      negative = 0;
      ++s;
    }
  else
    negative = 0;

  if (base == 16
      && s[0] == '0'
      && (s[1] == 'x' || s[1] == 'X'))
    s += 2;

  /* If BASE is zero, figure it out ourselves.  */
  if (base == 0)
    if (*s == '0')
      {
	if (s[1] == 'x' || s[1] == 'X')
	  {
	    s += 2;
	    base = 16;
	  }
	else
	  base = 8;
      }
    else
      base = 10;

  /* Save the pointer so we can check later if anything happened.  */
  save = s;

  cutoff = ULONG_MAX / (unsigned long int) base;
  cutlim = ULONG_MAX % (unsigned long int) base;

  overflow = 0;
  i = 0;
  for (c = BUCHAR (*s); c != '\0'; c = BUCHAR (*++s))
    {
      if (isdigit(c))
	c -= '0';
      else if (islower(c))
	c = c - 'a' + 10;
      else if (isupper(c))
	c = c - 'A' + 10;
      else
	break;
      if (c >= base)
	break;
      /* Check for overflow.  */
      if (i > cutoff || (i == cutoff && c > cutlim))
	overflow = 1;
      else
	{
	  i *= (unsigned long int) base;
	  i += c;
	}
    }

  /* Check if anything actually happened.  */
  if (s == save)
    goto noconv;

  /* Store in ENDPTR the address of one character
     past the last character we converted.  */
  if (endptr != NULL)
    *endptr = (char *) s;

#if	!UNSIGNED
  /* Check for a value that is within the range of
     `unsigned long int', but outside the range of `long int'.  */
  if (i > (negative ?
	   - (unsigned long int) LONG_MIN : (unsigned long int) LONG_MAX))
    overflow = 1;
#endif

  if (overflow)
    {
      errno = ERANGE;
#if	UNSIGNED
      return ULONG_MAX;
#else
      return negative ? LONG_MIN : LONG_MAX;
#endif
    }

  /* Return the result of the appropriate sign.  */
  return (negative ? - i : i);

 noconv:
  /* There was no number to convert.  */
  if (endptr != NULL)
    *endptr = (char *) nptr;
  return 0L;
}
