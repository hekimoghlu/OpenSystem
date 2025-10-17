/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 1, 2023.
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
#include <stdlib.h>
#include <string.h>
#include "secure.h"

char *
__stpncpy_chk (char *restrict dest, char *restrict src,
	       size_t len, size_t dstlen)
{
  size_t n;
  char *retval;

  if (__builtin_expect (dstlen < len, 0))
    __chk_fail_overflow ();

  retval = stpncpy (dest, src, len); // Normally returns a pointer to the \0
  n = retval - dest + 1;

  // Check if it's pointing to the location after the buffer after not writing \0
  if (n == len + 1)
    n--;

  if (__builtin_expect (__chk_assert_no_overlap != 0, 1))
    __chk_overlap(dest, n, src, n);

  return retval;
}
