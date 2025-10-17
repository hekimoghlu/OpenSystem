/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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

#if !defined(CURL_DISABLE_COOKIES) || !defined(CURL_DISABLE_ALTSVC) ||  \
  !defined(CURL_DISABLE_HSTS) || !defined(CURL_DISABLE_NETRC)

#include "curl_get_line.h"
#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

/*
 * Curl_get_line() makes sure to only return complete whole lines that end
 * newlines.
 */
int Curl_get_line(struct dynbuf *buf, FILE *input)
{
  CURLcode result;
  char buffer[128];
  Curl_dyn_reset(buf);
  while(1) {
    char *b = fgets(buffer, sizeof(buffer), input);

    if(b) {
      size_t rlen = strlen(b);

      if(!rlen)
        break;

      result = Curl_dyn_addn(buf, b, rlen);
      if(result)
        /* too long line or out of memory */
        return 0; /* error */

      else if(b[rlen-1] == '\n')
        /* end of the line */
        return 1; /* all good */

      else if(feof(input)) {
        /* append a newline */
        result = Curl_dyn_addn(buf, "\n", 1);
        if(result)
          /* too long line or out of memory */
          return 0; /* error */
        return 1; /* all good */
      }
    }
    else
      break;
  }
  return 0;
}

#endif /* if not disabled */
