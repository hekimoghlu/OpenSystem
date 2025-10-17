/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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

#if !defined(CURL_DISABLE_IMAP) || !defined(CURL_DISABLE_SMTP) ||       \
  !defined(CURL_DISABLE_POP3) || \
  (!defined(CURL_DISABLE_LDAP) && defined(USE_OPENLDAP))

#include <curl/curl.h>
#include "urldata.h"

#include "vauth/vauth.h"
#include "warnless.h"
#include "strtok.h"
#include "sendf.h"
#include "curl_printf.h"

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

/*
 * Curl_auth_create_plain_message()
 *
 * This is used to generate an already encoded PLAIN message ready
 * for sending to the recipient.
 *
 * Parameters:
 *
 * authzid [in]     - The authorization identity.
 * authcid [in]     - The authentication identity.
 * passwd  [in]     - The password.
 * out     [out]    - The result storage.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_plain_message(const char *authzid,
                                        const char *authcid,
                                        const char *passwd,
                                        struct bufref *out)
{
  char *plainauth;
  size_t plainlen;
  size_t zlen;
  size_t clen;
  size_t plen;

  zlen = (authzid == NULL ? 0 : strlen(authzid));
  clen = strlen(authcid);
  plen = strlen(passwd);

  /* Compute binary message length. Check for overflows. */
  if((zlen > SIZE_T_MAX/4) || (clen > SIZE_T_MAX/4) ||
     (plen > (SIZE_T_MAX/2 - 2)))
    return CURLE_OUT_OF_MEMORY;
  plainlen = zlen + clen + plen + 2;

  plainauth = malloc(plainlen + 1);
  if(!plainauth)
    return CURLE_OUT_OF_MEMORY;

  /* Calculate the reply */
  if(zlen)
    memcpy(plainauth, authzid, zlen);
  plainauth[zlen] = '\0';
  memcpy(plainauth + zlen + 1, authcid, clen);
  plainauth[zlen + clen + 1] = '\0';
  memcpy(plainauth + zlen + clen + 2, passwd, plen);
  plainauth[plainlen] = '\0';
  Curl_bufref_set(out, plainauth, plainlen, curl_free);
  return CURLE_OK;
}

/*
 * Curl_auth_create_login_message()
 *
 * This is used to generate an already encoded LOGIN message containing the
 * user name or password ready for sending to the recipient.
 *
 * Parameters:
 *
 * valuep  [in]     - The user name or user's password.
 * out     [out]    - The result storage.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_login_message(const char *valuep, struct bufref *out)
{
  Curl_bufref_set(out, valuep, strlen(valuep), NULL);
  return CURLE_OK;
}

/*
 * Curl_auth_create_external_message()
 *
 * This is used to generate an already encoded EXTERNAL message containing
 * the user name ready for sending to the recipient.
 *
 * Parameters:
 *
 * user    [in]     - The user name.
 * out     [out]    - The result storage.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_external_message(const char *user,
                                           struct bufref *out)
{
  /* This is the same formatting as the login message */
  return Curl_auth_create_login_message(user, out);
}

#endif /* if no users */
