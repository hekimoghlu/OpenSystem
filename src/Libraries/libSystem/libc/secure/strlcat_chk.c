/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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

size_t
__strlcat_chk(char *restrict dest, char *restrict src,
    size_t len, size_t dstlen)
{
  size_t initial_srclen;
  size_t initial_dstlen;

  if (__builtin_expect (dstlen < len, 0)) {
    __chk_fail_overflow ();
  }

  initial_srclen = strlen(src);
  initial_dstlen = strnlen(dest, len);

  if (initial_dstlen == len)
    return len+initial_srclen;

  if (initial_srclen < len - initial_dstlen) {
    __chk_overlap(dest, initial_srclen + initial_dstlen + 1, src, initial_srclen + 1);
    memcpy(dest+initial_dstlen, src, initial_srclen + 1);
  } else {
    __chk_overlap(dest, len, src, len - initial_dstlen - 1);
    memcpy(dest+initial_dstlen, src, len - initial_dstlen - 1);
    dest[len-1] = '\0';
  }

  return initial_srclen + initial_dstlen;
}
