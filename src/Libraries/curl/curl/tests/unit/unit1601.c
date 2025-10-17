/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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

#include "curl_md5.h"

static CURLcode unit_setup(void)
{
  return CURLE_OK;
}

static void unit_stop(void)
{

}

UNITTEST_START

#if (defined(USE_CURL_NTLM_CORE) && !defined(USE_WINDOWS_SSPI)) \
    || !defined(CURL_DISABLE_DIGEST_AUTH)

  const char string1[] = "1";
  const char string2[] = "hello-you-fool";
  unsigned char output[MD5_DIGEST_LEN];
  unsigned char *testp = output;

  Curl_md5it(output, (const unsigned char *) string1, strlen(string1));

  verify_memory(testp, "\xc4\xca\x42\x38\xa0\xb9\x23\x82\x0d\xcc\x50\x9a\x6f"
                "\x75\x84\x9b", MD5_DIGEST_LEN);

  Curl_md5it(output, (const unsigned char *) string2, strlen(string2));

  verify_memory(testp, "\x88\x67\x0b\x6d\x5d\x74\x2f\xad\xa5\xcd\xf9\xb6\x82"
                "\x87\x5f\x22", MD5_DIGEST_LEN);
#endif


UNITTEST_STOP
