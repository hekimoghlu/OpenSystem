/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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

void *
__memccpy_chk (void *dest, const void *src, int c, size_t len, size_t dstlen)
{
  void *retval;

  if (__builtin_expect (dstlen < len, 0))
    __chk_fail_overflow ();

  /* retval is NULL if len was copied, otherwise retval is the
   * byte *after* the last one written.
   */
  retval = memccpy (dest, src, c, len);

  if (retval != NULL) {
    len = (uintptr_t)retval - (uintptr_t)dest;
  }

  __chk_overlap(dest, len, src, len);

  return retval;
}
