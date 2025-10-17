/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
#include "altsvc.h"

static CURLcode
unit_setup(void)
{
  return CURLE_OK;
}

static void
unit_stop(void)
{
  curl_global_cleanup();
}

UNITTEST_START
#if !defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_ALTSVC)
{
  char outname[256];
  CURL *curl;
  CURLcode result;
  struct altsvcinfo *asi = Curl_altsvc_init();
  abort_if(!asi, "Curl_altsvc_i");
  result = Curl_altsvc_load(asi, arg);
  if(result) {
    fail_if(result, "Curl_altsvc_load");
    goto fail;
  }
  curl_global_init(CURL_GLOBAL_ALL);
  curl = curl_easy_init();
  if(!curl) {
    fail_if(!curl, "curl_easy_init");
    goto fail;
  }
  fail_unless(asi->list.size == 4, "wrong number of entries");
  msnprintf(outname, sizeof(outname), "%s-out", arg);

  result = Curl_altsvc_parse(curl, asi, "h2=\"example.com:8080\"\r\n",
                             ALPN_h1, "example.org", 8080);
  fail_if(result, "Curl_altsvc_parse() failed!");
  fail_unless(asi->list.size == 5, "wrong number of entries");

  result = Curl_altsvc_parse(curl, asi, "h3=\":8080\"\r\n",
                             ALPN_h1, "2.example.org", 8080);
  fail_if(result, "Curl_altsvc_parse(2) failed!");
  fail_unless(asi->list.size == 6, "wrong number of entries");

  result = Curl_altsvc_parse(curl, asi,
                             "h2=\"example.com:8080\", h3=\"yesyes.com\"\r\n",
                             ALPN_h1, "3.example.org", 8080);
  fail_if(result, "Curl_altsvc_parse(3) failed!");
  /* that one should make two entries */
  fail_unless(asi->list.size == 8, "wrong number of entries");

  result = Curl_altsvc_parse(curl, asi,
                             "h2=\"example.com:443\"; ma = 120;\r\n",
                             ALPN_h2, "example.org", 80);
  fail_if(result, "Curl_altsvc_parse(4) failed!");
  fail_unless(asi->list.size == 9, "wrong number of entries");

  /* quoted 'ma' value */
  result = Curl_altsvc_parse(curl, asi,
                             "h2=\"example.net:443\"; ma=\"180\";\r\n",
                             ALPN_h2, "example.net", 80);
  fail_if(result, "Curl_altsvc_parse(4) failed!");
  fail_unless(asi->list.size == 10, "wrong number of entries");

  result =
    Curl_altsvc_parse(curl, asi,
                      "h2=\":443\", h3=\":443\"; ma = 120; persist = 1\r\n",
                      ALPN_h1, "curl.se", 80);
  fail_if(result, "Curl_altsvc_parse(5) failed!");
  fail_unless(asi->list.size == 12, "wrong number of entries");

  /* clear that one again and decrease the counter */
  result = Curl_altsvc_parse(curl, asi, "clear;\r\n",
                             ALPN_h1, "curl.se", 80);
  fail_if(result, "Curl_altsvc_parse(6) failed!");
  fail_unless(asi->list.size == 10, "wrong number of entries");

  Curl_altsvc_save(curl, asi, outname);

  curl_easy_cleanup(curl);
fail:
  Curl_altsvc_cleanup(&asi);
}
#endif
UNITTEST_STOP
