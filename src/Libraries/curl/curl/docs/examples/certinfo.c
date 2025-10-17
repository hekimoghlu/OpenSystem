/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
 * Extract lots of TLS certificate info.
 * </DESC>
 */
#include <stdio.h>

#include <curl/curl.h>

static size_t wrfu(void *ptr,  size_t  size,  size_t  nmemb,  void *stream)
{
  (void)stream;
  (void)ptr;
  return size * nmemb;
}

int main(void)
{
  CURL *curl;
  CURLcode res;

  curl_global_init(CURL_GLOBAL_DEFAULT);

  curl = curl_easy_init();
  if(curl) {
    curl_easy_setopt(curl, CURLOPT_URL, "https://www.example.com/");

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, wrfu);

    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

    curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);
    curl_easy_setopt(curl, CURLOPT_CERTINFO, 1L);

    res = curl_easy_perform(curl);

    if(!res) {
      struct curl_certinfo *certinfo;

      res = curl_easy_getinfo(curl, CURLINFO_CERTINFO, &certinfo);

      if(!res && certinfo) {
        int i;

        printf("%d certs!\n", certinfo->num_of_certs);

        for(i = 0; i < certinfo->num_of_certs; i++) {
          struct curl_slist *slist;

          for(slist = certinfo->certinfo[i]; slist; slist = slist->next)
            printf("%s\n", slist->data);

        }
      }

    }

    curl_easy_cleanup(curl);
  }

  curl_global_cleanup();

  return 0;
}
