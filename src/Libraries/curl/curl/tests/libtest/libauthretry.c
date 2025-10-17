/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
/* argv1 = URL
 * argv2 = main auth type
 * argv3 = second auth type
 */

#include "test.h"
#include "memdebug.h"

static CURLcode send_request(CURL *curl, const char *url, int seq,
                             long auth_scheme, const char *userpwd)
{
  CURLcode res;
  size_t len = strlen(url) + 4 + 1;
  char *full_url = malloc(len);
  if(!full_url) {
    fprintf(stderr, "Not enough memory for full url\n");
    return CURLE_OUT_OF_MEMORY;
  }

  msnprintf(full_url, len, "%s%04d", url, seq);
  fprintf(stderr, "Sending new request %d to %s with credential %s "
          "(auth %ld)\n", seq, full_url, userpwd, auth_scheme);
  test_setopt(curl, CURLOPT_URL, full_url);
  test_setopt(curl, CURLOPT_VERBOSE, 1L);
  test_setopt(curl, CURLOPT_HEADER, 1L);
  test_setopt(curl, CURLOPT_HTTPGET, 1L);
  test_setopt(curl, CURLOPT_USERPWD, userpwd);
  test_setopt(curl, CURLOPT_HTTPAUTH, auth_scheme);

  res = curl_easy_perform(curl);

test_cleanup:
  free(full_url);
  return res;
}

static CURLcode send_wrong_password(CURL *curl, const char *url, int seq,
                                    long auth_scheme)
{
    return send_request(curl, url, seq, auth_scheme, "testuser:wrongpass");
}

static CURLcode send_right_password(CURL *curl, const char *url, int seq,
                                    long auth_scheme)
{
    return send_request(curl, url, seq, auth_scheme, "testuser:testpass");
}

static long parse_auth_name(const char *arg)
{
  if(!arg)
    return CURLAUTH_NONE;
  if(curl_strequal(arg, "basic"))
    return CURLAUTH_BASIC;
  if(curl_strequal(arg, "digest"))
    return CURLAUTH_DIGEST;
  if(curl_strequal(arg, "ntlm"))
    return CURLAUTH_NTLM;
  return CURLAUTH_NONE;
}

int test(char *url)
{
  CURLcode res;
  CURL *curl = NULL;

  long main_auth_scheme = parse_auth_name(libtest_arg2);
  long fallback_auth_scheme = parse_auth_name(libtest_arg3);

  if(main_auth_scheme == CURLAUTH_NONE ||
      fallback_auth_scheme == CURLAUTH_NONE) {
    fprintf(stderr, "auth schemes not found on commandline\n");
    return TEST_ERR_MAJOR_BAD;
  }

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  /* Send wrong password, then right password */

  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }

  res = send_wrong_password(curl, url, 100, main_auth_scheme);
  if(res != CURLE_OK)
    goto test_cleanup;

  res = send_right_password(curl, url, 200, fallback_auth_scheme);
  if(res != CURLE_OK)
    goto test_cleanup;

  curl_easy_cleanup(curl);

  /* Send wrong password twice, then right password */
  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }

  res = send_wrong_password(curl, url, 300, main_auth_scheme);
  if(res != CURLE_OK)
    goto test_cleanup;

  res = send_wrong_password(curl, url, 400, fallback_auth_scheme);
  if(res != CURLE_OK)
    goto test_cleanup;

  res = send_right_password(curl, url, 500, fallback_auth_scheme);
  if(res != CURLE_OK)
    goto test_cleanup;

test_cleanup:

  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return (int)res;
}
