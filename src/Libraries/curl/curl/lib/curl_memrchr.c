/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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

#include <curl/curl.h>

#include "curl_memrchr.h"
#include "curl_memory.h"

/* The last #include file should be: */
#include "memdebug.h"

#ifndef HAVE_MEMRCHR

/*
 * Curl_memrchr()
 *
 * Our memrchr() function clone for systems which lack this function. The
 * memrchr() function is like the memchr() function, except that it searches
 * backwards from the end of the n bytes pointed to by s instead of forward
 * from the beginning.
 */

void *
Curl_memrchr(const void *s, int c, size_t n)
{
  if(n > 0) {
    const unsigned char *p = s;
    const unsigned char *q = s;

    p += n - 1;

    while(p >= q) {
      if(*p == (unsigned char)c)
        return (void *)p;
      p--;
    }
  }
  return NULL;
}

#endif /* HAVE_MEMRCHR */
