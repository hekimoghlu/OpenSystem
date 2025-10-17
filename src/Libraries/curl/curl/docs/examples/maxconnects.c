/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 24, 2023.
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
 * Set maximum number of persistent connections to 1.
 * </DESC>
 */
#include <stdio.h>
#include <curl/curl.h>

int main(void)
{
  CURL *curl;
  CURLcode res;

  curl = curl_easy_init();
  if(curl) {
    const char *urls[] = { "https://example.com",
      "https://curl.se",
      "https://www.example/",
      NULL /* end of list */
    };
    int i = 0;

    /* Change the maximum number of persistent connection   */
    curl_easy_setopt(curl, CURLOPT_MAXCONNECTS, 1L);

    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

    /* loop over the URLs */
    while(urls[i]) {
      curl_easy_setopt(curl, CURLOPT_URL, urls[i]);

      /* Perform the request, res gets the return code */
      res = curl_easy_perform(curl);
      /* Check for errors */
      if(res != CURLE_OK)
        fprintf(stderr, "curl_easy_perform() failed: %s\n",
                curl_easy_strerror(res));
      i++;
    }
    /* always cleanup */
    curl_easy_cleanup(curl);
  }
  return 0;
}
