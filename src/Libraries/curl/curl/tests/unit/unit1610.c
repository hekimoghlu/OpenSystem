/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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

#include "curl_sha256.h"

static CURLcode unit_setup(void)
{
  CURLcode res = CURLE_OK;
  global_init(CURL_GLOBAL_ALL);
  return res;
}

static void unit_stop(void)
{
  curl_global_cleanup();
}

UNITTEST_START

#if !defined(CURL_DISABLE_AWS) || !defined(CURL_DISABLE_DIGEST_AUTH) \
    || defined(USE_LIBSSH2)

  const char string1[] = "1";
  const char string2[] = "hello-you-fool";
  unsigned char output[SHA256_DIGEST_LENGTH];
  unsigned char *testp = output;

  Curl_sha256it(output, (const unsigned char *) string1, strlen(string1));

  verify_memory(testp,
                "\x6b\x86\xb2\x73\xff\x34\xfc\xe1\x9d\x6b\x80\x4e\xff\x5a\x3f"
                "\x57\x47\xad\xa4\xea\xa2\x2f\x1d\x49\xc0\x1e\x52\xdd\xb7\x87"
                "\x5b\x4b", SHA256_DIGEST_LENGTH);

  Curl_sha256it(output, (const unsigned char *) string2, strlen(string2));

  verify_memory(testp,
                "\xcb\xb1\x6a\x8a\xb9\xcb\xb9\x35\xa8\xcb\xa0\x2e\x28\xc0\x26"
                "\x30\xd1\x19\x9c\x1f\x02\x17\xf4\x7c\x96\x20\xf3\xef\xe8\x27"
                "\x15\xae", SHA256_DIGEST_LENGTH);
#endif


UNITTEST_STOP
