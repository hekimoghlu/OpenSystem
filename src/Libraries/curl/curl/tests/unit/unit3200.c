/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
#include "curl_get_line.h"

#if !defined(CURL_DISABLE_COOKIES) || !defined(CURL_DISABLE_ALTSVC) ||  \
  !defined(CURL_DISABLE_HSTS) || !defined(CURL_DISABLE_NETRC)

/* The test XML does not supply a way to write files without newlines
 * so we write our own
 */

#define C64 "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
#define C256 C64 C64 C64 C64
#define C1024 C256 C256 C256 C256
#define C4096 C1024 C1024 C1024 C1024

static CURLcode unit_setup(void)
{
  return CURLE_OK;
}

static CURLcode unit_stop(void)
{
  return CURLE_OK;
}

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define NUMTESTS 6
static const char *filecontents[] = {
  /* Both should be read */
  "LINE1\n"
  "LINE2 NEWLINE\n",

  /* Both should be read */
  "LINE1\n"
  "LINE2 NONEWLINE",

  /* Only first should be read */
  "LINE1\n"
  C4096,

  /* First line should be read */
  "LINE1\n"
  C4096 "SOME EXTRA TEXT",

  /* Only first should be read */
  "LINE1\n"
  C4096 "SOME EXTRA TEXT\n"
  "LINE3\n",

  "LINE1\x1aTEST"
};

#ifdef __GNUC__
#pragma GCC diagnostic warning "-Woverlength-strings"
#endif


UNITTEST_START
  size_t i;
  int rc = 0;
  for(i = 0; i < NUMTESTS; i++) {
    FILE *fp;
    struct dynbuf buf;
    int len = 4096;
    char *line;
    Curl_dyn_init(&buf, len);

    fp = fopen(arg, "wb");
    abort_unless(fp != NULL, "Cannot open testfile");
    fwrite(filecontents[i], 1, strlen(filecontents[i]), fp);
    fclose(fp);

    fp = fopen(arg, "rb");
    abort_unless(fp != NULL, "Cannot open testfile");

    fprintf(stderr, "Test %zd...", i);
    switch(i) {
      case 0:
        rc = Curl_get_line(&buf, fp);
        line = Curl_dyn_ptr(&buf);
        fail_unless(line && !strcmp("LINE1\n", line),
                    "First line failed (1)");
        rc = Curl_get_line(&buf, fp);
        line = Curl_dyn_ptr(&buf);
        fail_unless(line && !strcmp("LINE2 NEWLINE\n", line),
                    "Second line failed (1)");
        rc = Curl_get_line(&buf, fp);
        abort_unless(!Curl_dyn_len(&buf), "Missed EOF (1)");
        break;
      case 1:
        rc = Curl_get_line(&buf, fp);
        line = Curl_dyn_ptr(&buf);
        fail_unless(line && !strcmp("LINE1\n", line),
                    "First line failed (2)");
        rc = Curl_get_line(&buf, fp);
        line = Curl_dyn_ptr(&buf);
        fail_unless(line && !strcmp("LINE2 NONEWLINE\n", line),
                    "Second line failed (2)");
        rc = Curl_get_line(&buf, fp);
        abort_unless(!Curl_dyn_len(&buf), "Missed EOF (2)");
        break;
      case 2:
        rc = Curl_get_line(&buf, fp);
        line = Curl_dyn_ptr(&buf);
        fail_unless(line && !strcmp("LINE1\n", line),
                    "First line failed (3)");
        rc = Curl_get_line(&buf, fp);
        fail_unless(!Curl_dyn_len(&buf),
                    "Did not detect max read on EOF (3)");
        break;
      case 3:
        rc = Curl_get_line(&buf, fp);
        line = Curl_dyn_ptr(&buf);
        fail_unless(line && !strcmp("LINE1\n", line),
                    "First line failed (4)");
        rc = Curl_get_line(&buf, fp);
        fail_unless(!Curl_dyn_len(&buf),
                    "Did not ignore partial on EOF (4)");
        break;
      case 4:
        rc = Curl_get_line(&buf, fp);
        line = Curl_dyn_ptr(&buf);
        fail_unless(line && !strcmp("LINE1\n", line),
                    "First line failed (5)");
        rc = Curl_get_line(&buf, fp);
        fail_unless(!Curl_dyn_len(&buf),
                    "Did not bail out on too long line");
        break;
      case 5:
        rc = Curl_get_line(&buf, fp);
        line = Curl_dyn_ptr(&buf);
        fail_unless(line && !strcmp("LINE1\x1aTEST\n", line),
                    "Missed/Misinterpreted ^Z (6)");
        rc = Curl_get_line(&buf, fp);
        abort_unless(!Curl_dyn_len(&buf), "Missed EOF (6)");
        break;
      default:
        abort_unless(1, "Unknown case");
        break;
    }
    Curl_dyn_free(&buf);
    fclose(fp);
    fprintf(stderr, "OK\n");
  }
  return rc;
UNITTEST_STOP

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#else
static CURLcode unit_setup(void)
{
  return CURLE_OK;
}
static void unit_stop(void)
{
}
UNITTEST_START
UNITTEST_STOP

#endif
