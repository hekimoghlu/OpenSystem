/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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

#if !defined(CURL_DISABLE_IMAP) || !defined(CURL_DISABLE_SMTP) || \
  !defined(CURL_DISABLE_POP3) || \
  (!defined(CURL_DISABLE_LDAP) && defined(USE_OPENLDAP))

#include <curl/curl.h>
#include "urldata.h"

#include "vauth/vauth.h"
#include "warnless.h"
#include "curl_printf.h"

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

/*
 * Curl_auth_create_oauth_bearer_message()
 *
 * This is used to generate an OAuth 2.0 message ready for sending to the
 * recipient.
 *
 * Parameters:
 *
 * user[in]         - The user name.
 * host[in]         - The host name.
 * port[in]         - The port(when not Port 80).
 * bearer[in]       - The bearer token.
 * out[out]         - The result storage.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_oauth_bearer_message(const char *user,
                                               const char *host,
                                               const long port,
                                               const char *bearer,
                                               struct bufref *out)
{
  char *oauth;

  /* Generate the message */
  if(port == 0 || port == 80)
    oauth = aprintf("n,a=%s,\1host=%s\1auth=Bearer %s\1\1", user, host,
                    bearer);
  else
    oauth = aprintf("n,a=%s,\1host=%s\1port=%ld\1auth=Bearer %s\1\1", user,
                    host, port, bearer);
  if(!oauth)
    return CURLE_OUT_OF_MEMORY;

  Curl_bufref_set(out, oauth, strlen(oauth), curl_free);
  return CURLE_OK;
}

/*
 * Curl_auth_create_xoauth_bearer_message()
 *
 * This is used to generate a XOAuth 2.0 message ready for * sending to the
 * recipient.
 *
 * Parameters:
 *
 * user[in]         - The user name.
 * bearer[in]       - The bearer token.
 * out[out]         - The result storage.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_xoauth_bearer_message(const char *user,
                                               const char *bearer,
                                               struct bufref *out)
{
  /* Generate the message */
  char *xoauth = aprintf("user=%s\1auth=Bearer %s\1\1", user, bearer);
  if(!xoauth)
    return CURLE_OUT_OF_MEMORY;

  Curl_bufref_set(out, xoauth, strlen(xoauth), curl_free);
  return CURLE_OK;
}
#endif /* disabled, no users */
