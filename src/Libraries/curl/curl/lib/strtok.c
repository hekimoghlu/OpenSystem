/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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
#include "curl_setup.h"

#ifndef HAVE_STRTOK_R
#include <stddef.h>

#include "strtok.h"

char *
Curl_strtok_r(char *ptr, const char *sep, char **end)
{
  if(!ptr)
    /* we got NULL input so then we get our last position instead */
    ptr = *end;

  /* pass all letters that are including in the separator string */
  while(*ptr && strchr(sep, *ptr))
    ++ptr;

  if(*ptr) {
    /* so this is where the next piece of string starts */
    char *start = ptr;

    /* set the end pointer to the first byte after the start */
    *end = start + 1;

    /* scan through the string to find where it ends, it ends on a
       null byte or a character that exists in the separator string */
    while(**end && !strchr(sep, **end))
      ++*end;

    if(**end) {
      /* the end is not a null byte */
      **end = '\0';  /* null-terminate it! */
      ++*end;        /* advance the last pointer to beyond the null byte */
    }

    return start; /* return the position where the string starts */
  }

  /* we ended up on a null byte, there are no more strings to find! */
  return NULL;
}

#endif /* this was only compiled if strtok_r wasn't present */

