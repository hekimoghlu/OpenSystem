/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"
#include "addrtostr.h"

#include <stdio.h>
#include <string.h>

/*
 *
 */

#ifndef IN6ADDRSZ
#define IN6ADDRSZ   16   /* IPv6 T_AAAA */
#endif

#ifndef INT16SZ
#define INT16SZ     2    /* word size */
#endif

const char *
addrtostr (const void *src, char *dst, size_t size)
{
    const u_char *srcaddr = (const u_char *)src;
    const char digits[] = "0123456789";
    int i;
    const char *orig_dst = dst;

    if (size < INET_ADDRSTRLEN) {
	errno = ENOSPC;
	return NULL;
    }
    for (i = 0; i < 4; ++i) {
    	int n = *srcaddr++;
	int non_zerop = 0;

	if (non_zerop || n / 100 > 0) {
	    *dst++ = digits[n / 100];
	    n %= 100;
	    non_zerop = 1;
	}
	if (non_zerop || n / 10 > 0) {
	    *dst++ = digits[n / 10];
	    n %= 10;
	    non_zerop = 1;
	}
	*dst++ = digits[n];
	if (i != 3)
	    *dst++ = '.';
    }
    *dst++ = '\0';
    return orig_dst;
}

/*
 * Convert IPv6 binary address into presentation (printable) format.
 */
const char *
addrtostr6 (const void *src, char *dst, size_t size)
{
  /*
   * Note that int32_t and int16_t need only be "at least" large enough
   * to contain a value of the specified size.  On some systems, like
   * Crays, there is no such thing as an integer variable with 16 bits.
   * Keep this in mind if you think this function should have been coded
   * to use pointer overlays.  All the world's not a VAX.
   */
  const u_char *srcaddr = (const u_char *)src;
  char *dp;
  size_t space_left, added_space;
  int snprintfed;
  struct {
    int base;
    int len;
  } best, cur;
  uint16_t words [IN6ADDRSZ / INT16SZ];
  int  i;

  /* Preprocess:
   *  Copy the input (bytewise) array into a wordwise array.
   *  Find the longest run of 0x00's in src[] for :: shorthanding.
   */
  for (i = 0; i < (IN6ADDRSZ / INT16SZ); i++)
      words[i] = (srcaddr[2*i] << 8) | srcaddr[2*i + 1];

  best.len = 0;
  best.base = -1;
  cur.len = 0;
  cur.base  = -1;
  for (i = 0; i < (IN6ADDRSZ / INT16SZ); i++)
  {
    if (words[i] == 0)
    {
      if (cur.base == -1)
           cur.base = i, cur.len = 1;
      else cur.len++;
    }
    else if (cur.base != -1)
    {
      if (best.base == -1 || cur.len > best.len)
         best = cur;
      cur.base = -1;
    }
  }
  if ((cur.base != -1) && (best.base == -1 || cur.len > best.len))
     best = cur;
  if (best.base != -1 && best.len < 2)
     best.base = -1;

  /* Format the result.
   */
  dp = dst;
  space_left = size;
#define APPEND_CHAR(c) \
    { \
        if (space_left == 0) { \
            errno = ENOSPC; \
            return (NULL); \
        } \
        *dp++ = c; \
        space_left--; \
    }
  for (i = 0; i < (IN6ADDRSZ / INT16SZ); i++)
  {
    /* Are we inside the best run of 0x00's?
     */
    if (best.base != -1 && i >= best.base && i < (best.base + best.len))
    {
      if (i == best.base)
      	 APPEND_CHAR(':');
      continue;
    }

    /* Are we following an initial run of 0x00s or any real hex?
     */
    if (i != 0)
       APPEND_CHAR(':');

    /* Is this address an encapsulated IPv4?
     */
    if (i == 6 && best.base == 0 &&
        (best.len == 6 || (best.len == 5 && words[5] == 0xffff)))
    {
      if (!addrtostr(srcaddr+12, dp, space_left))
      {
        errno = ENOSPC;
        return (NULL);
      }
      added_space = strlen(dp);
      dp += added_space;
      space_left -= added_space;
      break;
    }
    snprintfed = snprintf (dp, space_left, "%x", words[i]);
    if (snprintfed < 0)
        return (NULL);
    if ((size_t) snprintfed >= space_left)
    {
        errno = ENOSPC;
        return (NULL);
    }
    dp += snprintfed;
    space_left -= snprintfed;
  }

  /* Was it a trailing run of 0x00's?
   */
  if (best.base != -1 && (best.base + best.len) == (IN6ADDRSZ / INT16SZ))
     APPEND_CHAR(':');
  APPEND_CHAR('\0');

  return (dst);
}
