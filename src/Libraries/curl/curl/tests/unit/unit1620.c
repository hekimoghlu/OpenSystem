/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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

#include "urldata.h"
#include "url.h"

#include "memdebug.h" /* LAST include file */

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
{
  int rc;
  struct Curl_easy *empty;
  const char *hostname = "hostname";
  enum dupstring i;

  bool async = FALSE;
  bool protocol_connect = FALSE;

  rc = Curl_open(&empty);
  if(rc)
    goto unit_test_abort;
  fail_unless(rc == CURLE_OK, "Curl_open() failed");

  rc = Curl_connect(empty, &async, &protocol_connect);
  fail_unless(rc == CURLE_URL_MALFORMAT,
              "Curl_connect() failed to return CURLE_URL_MALFORMAT");

  fail_unless(empty->magic == CURLEASY_MAGIC_NUMBER,
              "empty->magic should be equal to CURLEASY_MAGIC_NUMBER");

  /* double invoke to ensure no dependency on internal state */
  rc = Curl_connect(empty, &async, &protocol_connect);
  fail_unless(rc == CURLE_URL_MALFORMAT,
              "Curl_connect() failed to return CURLE_URL_MALFORMAT");

  rc = Curl_init_userdefined(empty);
  fail_unless(rc == CURLE_OK, "Curl_userdefined() failed");

  rc = Curl_init_do(empty, empty->conn);
  fail_unless(rc == CURLE_OK, "Curl_init_do() failed");

  rc = Curl_parse_login_details(
                          hostname, strlen(hostname), NULL, NULL, NULL);
  fail_unless(rc == CURLE_OK,
              "Curl_parse_login_details() failed");

  Curl_freeset(empty);
  for(i = (enum dupstring)0; i < STRING_LAST; i++) {
    fail_unless(empty->set.str[i] == NULL,
                "Curl_free() did not set to NULL");
  }

  rc = Curl_close(&empty);
  fail_unless(rc == CURLE_OK, "Curl_close() failed");

}
UNITTEST_STOP
