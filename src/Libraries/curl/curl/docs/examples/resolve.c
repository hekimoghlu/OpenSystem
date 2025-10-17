/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
 * Use CURLOPT_RESOLVE to feed custom IP addresses for given hostname + port
 * number combinations.
 * </DESC>
 */
#include <stdio.h>
#include <curl/curl.h>

int main(void)
{
  CURL *curl;
  CURLcode res = CURLE_OK;

  /* Each single name resolve string should be written using the format
     HOST:PORT:ADDRESS where HOST is the name libcurl tries to resolve, PORT
     is the port number of the service where libcurl wants to connect to the
     HOST and ADDRESS is the numerical IP address
   */
  struct curl_slist *host = curl_slist_append(NULL,
                                              "example.com:443:127.0.0.1");

  curl = curl_easy_init();
  if(curl) {
    curl_easy_setopt(curl, CURLOPT_RESOLVE, host);
    curl_easy_setopt(curl, CURLOPT_URL, "https://example.com");
    res = curl_easy_perform(curl);

    /* always cleanup */
    curl_easy_cleanup(curl);
  }

  curl_slist_free_all(host);

  return (int)res;
}
