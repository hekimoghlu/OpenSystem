/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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
#define CURL_NO_FMT_CHECKS

#include "curlcheck.h"

#include "curl/mprintf.h"

static CURLcode unit_setup(void) {return CURLE_OK;}
static void unit_stop(void) {}

UNITTEST_START

int rc;
char buf[3] = {'b', 'u', 'g'};
const char *str = "bug";
int width = 3;
char output[24];

/*#define curl_msnprintf snprintf */

/* without a trailing zero */
rc = curl_msnprintf(output, 4, "%.*s", width, buf);
fail_unless(rc == 3, "return code should be 3");
fail_unless(!strcmp(output, "bug"), "wrong output");

/* with a trailing zero */
rc = curl_msnprintf(output, 4, "%.*s", width, str);
fail_unless(rc == 3, "return code should be 3");
fail_unless(!strcmp(output, "bug"), "wrong output");

width = 2;
/* one byte less */
rc = curl_msnprintf(output, 4, "%.*s", width, buf);
fail_unless(rc == 2, "return code should be 2");
fail_unless(!strcmp(output, "bu"), "wrong output");

/* string with larger precision */
rc = curl_msnprintf(output, 8, "%.8s", str);
fail_unless(rc == 3, "return code should be 3");
fail_unless(!strcmp(output, "bug"), "wrong output");

/* longer string with precision */
rc = curl_msnprintf(output, 8, "%.3s", "0123456789");
fail_unless(rc == 3, "return code should be 3");
fail_unless(!strcmp(output, "012"), "wrong output");

/* negative width */
rc = curl_msnprintf(output, 8, "%-8s", str);
fail_unless(rc == 7, "return code should be 7");
fail_unless(!strcmp(output, "bug    "), "wrong output");

/* larger width that string length */
rc = curl_msnprintf(output, 8, "%8s", str);
fail_unless(rc == 7, "return code should be 7");
fail_unless(!strcmp(output, "     bu"), "wrong output");

/* output a number in a limited output */
rc = curl_msnprintf(output, 4, "%d", 10240);
fail_unless(rc == 3, "return code should be 3");
fail_unless(!strcmp(output, "102"), "wrong output");

/* padded strings */
rc = curl_msnprintf(output, 16, "%8s%8s", str, str);
fail_unless(rc == 15, "return code should be 15");
fail_unless(!strcmp(output, "     bug     bu"), "wrong output");

/* padded numbers */
rc = curl_msnprintf(output, 16, "%8d%8d", 1234, 5678);
fail_unless(rc == 15, "return code should be 15");
fail_unless(!strcmp(output, "    1234    567"), "wrong output");

/* double precision */
rc = curl_msnprintf(output, 24, "%2$.*1$.99d", 3, 5678);
fail_unless(rc == 0, "return code should be 0");

UNITTEST_STOP

