/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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
#include "curlcheck.h"

#include "urldata.h"
#include "url.h"

#include "memdebug.h" /* LAST include file */

static CURLcode unit_setup(void)
{
  return CURLE_OK;
}

static void unit_stop(void)
{
}

#if defined(__MINGW32__)  || \
  (!defined(HAVE_FSETXATTR) && \
  (!defined(__FreeBSD_version) || (__FreeBSD_version < 500000)))
UNITTEST_START
UNITTEST_STOP
#else

char *stripcredentials(const char *url);

struct checkthis {
  const char *input;
  const char *output;
};

static const struct checkthis tests[] = {
  { "ninja://foo@example.com", "ninja://foo@example.com" },
  { "https://foo@example.com", "https://example.com/" },
  { "https://localhost:45", "https://localhost:45/" },
  { "https://foo@localhost:45", "https://localhost:45/" },
  { "http://daniel:password@localhost", "http://localhost/" },
  { "http://daniel@localhost", "http://localhost/" },
  { "http://localhost/", "http://localhost/" },
  { NULL, NULL } /* end marker */
};

UNITTEST_START
{
  int i;

  for(i = 0; tests[i].input; i++) {
    const char *url = tests[i].input;
    char *stripped = stripcredentials(url);
    printf("Test %u got input \"%s\", output: \"%s\"\n",
           i, tests[i].input, stripped);

    fail_if(stripped && strcmp(tests[i].output, stripped),
            tests[i].output);
    curl_free(stripped);
  }
}
UNITTEST_STOP
#endif
