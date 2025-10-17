/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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
#include "test.h"

#define fail_unless(expr, msg)                             \
  do {                                                     \
    if(!(expr)) {                                          \
      fprintf(stderr, "%s:%d Assertion '%s' failed: %s\n", \
              __FILE__, __LINE__, #expr, msg);             \
      return 1;                                            \
    }                                                      \
  } while(0)

int test(char *URL)
{
  int rc;
  (void)URL;

  rc = curl_strequal("iii", "III");
  fail_unless(rc != 0, "return code should be non-zero");

  rc = curl_strequal("iiia", "III");
  fail_unless(rc == 0, "return code should be zero");

  rc = curl_strequal("iii", "IIIa");
  fail_unless(rc == 0, "return code should be zero");

  rc = curl_strequal("iiiA", "IIIa");
  fail_unless(rc != 0, "return code should be non-zero");

  rc = curl_strnequal("iii", "III", 3);
  fail_unless(rc != 0, "return code should be non-zero");

  rc = curl_strnequal("iiiABC", "IIIcba", 3);
  fail_unless(rc != 0, "return code should be non-zero");

  rc = curl_strnequal("ii", "II", 3);
  fail_unless(rc != 0, "return code should be non-zero");

  return 0;
}
