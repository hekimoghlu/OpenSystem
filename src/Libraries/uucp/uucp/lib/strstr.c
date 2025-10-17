/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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

/* Return the first ocurrence of NEEDLE in HAYSTACK.  */
char *
strstr (haystack, needle)
     const char *const haystack;
     const char *const needle;
{
  register const char *const needle_end = strchr(needle, '\0');
  register const char *const haystack_end = strchr(haystack, '\0');
  register const size_t needle_len = needle_end - needle;
  register const size_t needle_last = needle_len - 1;
  register const char *begin;

  if (needle_len == 0)
    return (char *) haystack_end;
  if ((size_t) (haystack_end - haystack) < needle_len)
    return NULL;

  for (begin = &haystack[needle_last]; begin < haystack_end; ++begin)
    {
      register const char *n = &needle[needle_last];
      register const char *h = begin;
      do
	if (*h != *n)
	  goto loop;
      while (--n >= needle && --h >= haystack);

      return (char *) h;
    loop:;
    }

  return NULL;
}
