/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
__snprintf_chk (char *dest, size_t len, int flags, size_t dstlen,
		const char *format, ...)
{
  va_list arg;
  int done;

  if (__builtin_expect (dstlen < len, 0))
    __chk_fail_overflow ();

  va_start (arg, format);

  done = vsnprintf (dest, len, format, arg);

  va_end (arg);

  return done;
}

int
__snprintf_object_size_chk (char *dest, size_t dstlen, size_t len,
const char *format, ...)
{
    va_list arg;
    int done;

    if (__builtin_expect (dstlen < len, 0))
        __chk_fail_overflow ();

    va_start (arg, format);

    done = vsnprintf (dest, len, format, arg);

    va_end (arg);

    return done;
}
