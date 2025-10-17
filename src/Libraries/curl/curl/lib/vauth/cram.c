/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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

#ifndef CURL_DISABLE_DIGEST_AUTH

#include <curl/curl.h>
#include "urldata.h"

#include "vauth/vauth.h"
#include "curl_hmac.h"
#include "curl_md5.h"
#include "warnless.h"
#include "curl_printf.h"

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"


/*
 * Curl_auth_create_cram_md5_message()
 *
 * This is used to generate a CRAM-MD5 response message ready for sending to
 * the recipient.
 *
 * Parameters:
 *
 * chlg    [in]     - The challenge.
 * userp   [in]     - The user name.
 * passwdp [in]     - The user's password.
 * out     [out]    - The result storage.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_cram_md5_message(const struct bufref *chlg,
                                           const char *userp,
                                           const char *passwdp,
                                           struct bufref *out)
{
  struct HMAC_context *ctxt;
  unsigned char digest[MD5_DIGEST_LEN];
  char *response;

  /* Compute the digest using the password as the key */
  ctxt = Curl_HMAC_init(Curl_HMAC_MD5,
                        (const unsigned char *) passwdp,
                        curlx_uztoui(strlen(passwdp)));
  if(!ctxt)
    return CURLE_OUT_OF_MEMORY;

  /* Update the digest with the given challenge */
  if(Curl_bufref_len(chlg))
    Curl_HMAC_update(ctxt, Curl_bufref_ptr(chlg),
                     curlx_uztoui(Curl_bufref_len(chlg)));

  /* Finalise the digest */
  Curl_HMAC_final(ctxt, digest);

  /* Generate the response */
  response = aprintf(
    "%s %02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
    userp, digest[0], digest[1], digest[2], digest[3], digest[4],
    digest[5], digest[6], digest[7], digest[8], digest[9], digest[10],
    digest[11], digest[12], digest[13], digest[14], digest[15]);
  if(!response)
    return CURLE_OUT_OF_MEMORY;

  Curl_bufref_set(out, response, strlen(response), curl_free);
  return CURLE_OK;
}

#endif /* !CURL_DISABLE_DIGEST_AUTH */
