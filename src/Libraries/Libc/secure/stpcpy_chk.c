/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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
__stpcpy_chk (char *dest, const char *src, size_t dstlen)
{
  char *retval = stpcpy(dest, src); // Returns a pointer to the \0
  size_t len = retval - dest + 1;

  if (__builtin_expect (dstlen < len, 0))
    __chk_fail_overflow ();

  if (__builtin_expect (__chk_assert_no_overlap != 0, 1))
    __chk_overlap(dest, len, src, len);

  return retval;
}
