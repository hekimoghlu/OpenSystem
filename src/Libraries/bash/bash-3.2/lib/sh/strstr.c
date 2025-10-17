/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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
 * My personal strstr() implementation that beats most other algorithms.
 * Until someone tells me otherwise, I assume that this is the
 * fastest implementation of strstr() in C.
 * I deliberately chose not to comment it.  You should have at least
 * as much fun trying to understand it, as I had to write it :-).
 *
 * Stephen R. van den Berg, berg@pool.informatik.rwth-aachen.de	*/

#if HAVE_CONFIG_H
# include <config.h>
#endif

#if defined _LIBC || defined HAVE_STRING_H
# include <string.h>
#endif
#include <sys/types.h>

typedef unsigned chartype;

#undef strstr

char *
strstr (const char *phaystack, const char *pneedle)
{
  register const unsigned char *haystack, *needle;
  register chartype b, c;

  haystack = (const unsigned char *) phaystack;
  needle = (const unsigned char *) pneedle;

  b = *needle;
  if (b != '\0')
    {
      haystack--;				/* possible ANSI violation */
      do
	{
	  c = *++haystack;
	  if (c == '\0')
	    goto ret0;
	}
      while (c != b);

      c = *++needle;
      if (c == '\0')
	goto foundneedle;
      ++needle;
      goto jin;

      for (;;)
        {
          register chartype a;
	  register const unsigned char *rhaystack, *rneedle;

	  do
	    {
	      a = *++haystack;
	      if (a == '\0')
		goto ret0;
	      if (a == b)
		break;
	      a = *++haystack;
	      if (a == '\0')
		goto ret0;
shloop:;    }
          while (a != b);

jin:	  a = *++haystack;
	  if (a == '\0')
	    goto ret0;

	  if (a != c)
	    goto shloop;

	  rhaystack = haystack-- + 1;
	  rneedle = needle;
	  a = *rneedle;

	  if (*rhaystack == a)
	    do
	      {
		if (a == '\0')
		  goto foundneedle;
		++rhaystack;
		a = *++needle;
		if (*rhaystack != a)
		  break;
		if (a == '\0')
		  goto foundneedle;
		++rhaystack;
		a = *++needle;
	      }
	    while (*rhaystack == a);

	  needle = rneedle;		   /* took the register-poor approach */

	  if (a == '\0')
	    break;
        }
    }
foundneedle:
  return (char*) haystack;
ret0:
  return 0;
}
