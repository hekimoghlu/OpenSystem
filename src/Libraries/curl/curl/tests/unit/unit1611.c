/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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

#include "curl_md4.h"

static CURLcode unit_setup(void)
{
  return CURLE_OK;
}

static void unit_stop(void)
{

}

UNITTEST_START

#ifdef USE_CURL_NTLM_CORE
  const char string1[] = "1";
  const char string2[] = "hello-you-fool";
  unsigned char output[MD4_DIGEST_LENGTH];
  unsigned char *testp = output;

  Curl_md4it(output, (const unsigned char *) string1, strlen(string1));

  verify_memory(testp,
                "\x8b\xe1\xec\x69\x7b\x14\xad\x3a\x53\xb3\x71\x43\x61\x20\x64"
                "\x1d", MD4_DIGEST_LENGTH);

  Curl_md4it(output, (const unsigned char *) string2, strlen(string2));

  verify_memory(testp,
                "\xa7\x16\x1c\xad\x7e\xbe\xdb\xbc\xf8\xc7\x23\x10\x2d\x2c\xe2"
                "\x0b", MD4_DIGEST_LENGTH);
#endif


UNITTEST_STOP
