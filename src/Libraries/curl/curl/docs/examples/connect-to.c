/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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
/* <DESC>
 * Use CURLOPT_CONNECT_TO to connect to "wrong" hostname
 * </DESC>
 */
#include <stdio.h>
#include <curl/curl.h>

int main(void)
{
  CURL *curl;
  CURLcode res = CURLE_OK;

  /*
    Each single string should be written using the format
    HOST:PORT:CONNECT-TO-HOST:CONNECT-TO-PORT where HOST is the host of the
    request, PORT is the port of the request, CONNECT-TO-HOST is the host name
    to connect to, and CONNECT-TO-PORT is the port to connect to.
   */
  /* instead of curl.se:443, it resolves and uses example.com:443 but in other
     aspects work as if it still is curl.se */
  struct curl_slist *host = curl_slist_append(NULL,
                                              "curl.se:443:example.com:443");

  curl = curl_easy_init();
  if(curl) {
    curl_easy_setopt(curl, CURLOPT_CONNECT_TO, host);
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
    curl_easy_setopt(curl, CURLOPT_URL, "https://curl.se/");

    /* since this connects to the wrong host, checking the host name in the
       server certificate fails, so unless we disable the check libcurl
       returns CURLE_PEER_FAILED_VERIFICATION */
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

    /* Letting the wrong host name in the certificate be okay, the transfer
       goes through but (most likely) causes a 404 or similar because it sends
       an unknown name in the Host: header field */
    res = curl_easy_perform(curl);

    /* always cleanup */
    curl_easy_cleanup(curl);
  }

  curl_slist_free_all(host);

  return (int)res;
}
