/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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
 * Extract headers post transfer with the header API
 * </DESC>
 */
#include <stdio.h>
#include <curl/curl.h>

static size_t write_cb(char *data, size_t n, size_t l, void *userp)
{
  /* take care of the data here, ignored in this example */
  (void)data;
  (void)userp;
  return n*l;
}

int main(void)
{
  CURL *curl;

  curl = curl_easy_init();
  if(curl) {
    CURLcode res;
    struct curl_header *header;
    curl_easy_setopt(curl, CURLOPT_URL, "https://example.com");
    /* example.com is redirected, so we tell libcurl to follow redirection */
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    /* this example just ignores the content */
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);

    /* Perform the request, res gets the return code */
    res = curl_easy_perform(curl);
    /* Check for errors */
    if(res != CURLE_OK)
      fprintf(stderr, "curl_easy_perform() failed: %s\n",
              curl_easy_strerror(res));

    if(CURLHE_OK == curl_easy_header(curl, "Content-Type", 0, CURLH_HEADER,
                                     -1, &header))
      printf("Got content-type: %s\n", header->value);

    printf("All server headers:\n");
    {
      struct curl_header *h;
      struct curl_header *prev = NULL;
      do {
        h = curl_easy_nextheader(curl, CURLH_HEADER, -1, prev);
        if(h)
          printf(" %s: %s (%u)\n", h->name, h->value, (int)h->amount);
        prev = h;
      } while(h);

    }
    /* always cleanup */
    curl_easy_cleanup(curl);
  }
  return 0;
}
