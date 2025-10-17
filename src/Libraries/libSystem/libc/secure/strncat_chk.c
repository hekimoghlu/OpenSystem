/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
__strncat_chk (char *restrict dest, const char *restrict append,
               size_t len, size_t dstlen)
{
  size_t len1 = strlen(dest);
  size_t len2 = strnlen(append, len);

  if (__builtin_expect (dstlen < len1 + len2 + 1, 0))
    __chk_fail_overflow ();

  if (__builtin_expect (__chk_assert_no_overlap != 0, 1))
    __chk_overlap(dest, len1 + len2 + 1, append, len2 + 1);

  /* memmove() all but the NUL, since it might not actually be NUL */
  memcpy(dest + len1, append, len2);
  dest[len1 + len2] = '\0';

  return dest;
}
