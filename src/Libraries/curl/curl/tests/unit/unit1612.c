/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 3, 2022.
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

#include "curl_hmac.h"
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

  const char password[] = "Pa55worD";
  const char string1[] = "1";
  const char string2[] = "hello-you-fool";
  unsigned char output[HMAC_MD5_LENGTH];
  unsigned char *testp = output;

  Curl_hmacit(Curl_HMAC_MD5,
              (const unsigned char *) password, strlen(password),
              (const unsigned char *) string1, strlen(string1),
              output);

  verify_memory(testp,
                "\xd1\x29\x75\x43\x58\xdc\xab\x78\xdf\xcd\x7f\x2b\x29\x31\x13"
                "\x37", HMAC_MD5_LENGTH);

  Curl_hmacit(Curl_HMAC_MD5,
              (const unsigned char *) password, strlen(password),
              (const unsigned char *) string2, strlen(string2),
              output);

  verify_memory(testp,
                "\x75\xf1\xa7\xb9\xf5\x40\xe5\xa4\x98\x83\x9f\x64\x5a\x27\x6d"
                "\xd0", HMAC_MD5_LENGTH);
#endif


UNITTEST_STOP
