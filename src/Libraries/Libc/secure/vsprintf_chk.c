/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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
#include <stdio.h>
#include <stdarg.h>
#include <limits.h>
#include "secure.h"

int
__vsprintf_chk (char *dest, int flags, size_t dstlen, const char *format,
		va_list arg)
{
  int done;

  if (__builtin_expect (dstlen > (size_t) INT_MAX, 0))
    done = vsprintf (dest, format, arg);
  else
    {
      done = vsnprintf (dest, dstlen, format, arg);
      if (__builtin_expect (done >= 0 && (size_t) done >= dstlen, 0))
        __chk_fail_overflow ();
    }

  return done;
}
