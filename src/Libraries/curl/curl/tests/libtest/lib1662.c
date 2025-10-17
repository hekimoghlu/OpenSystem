/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
#include "test.h"

static char data[]="mooaaa";

struct WriteThis {
  size_t sizeleft;
};

static size_t read_callback(char *ptr, size_t size, size_t nmemb, void *userp)
{
  struct WriteThis *pooh = (struct WriteThis *)userp;
  size_t len = strlen(data);

  if(size*nmemb < len)
    return 0;

  if(pooh->sizeleft) {
    memcpy(ptr, data, strlen(data));
    pooh->sizeleft = 0;
    return len;
  }

  return 0;                         /* no more data left to deliver */
}


int test(char *URL)
{
  CURLcode res = CURLE_OK;
  CURL *hnd;
  curl_mime *mime1;
  curl_mimepart *part1;
  struct WriteThis pooh = { 1 };

  mime1 = NULL;

  global_init(CURL_GLOBAL_ALL);

  hnd = curl_easy_init();
  if(hnd) {
    curl_easy_setopt(hnd, CURLOPT_BUFFERSIZE, 102400L);
    curl_easy_setopt(hnd, CURLOPT_URL, URL);
    curl_easy_setopt(hnd, CURLOPT_NOPROGRESS, 1L);
    mime1 = curl_mime_init(hnd);
    if(mime1) {
      part1 = curl_mime_addpart(mime1);
      curl_mime_data_cb(part1, -1, read_callback, NULL, NULL, &pooh);
      curl_mime_filename(part1, "poetry.txt");
      curl_mime_name(part1, "content");
      curl_easy_setopt(hnd, CURLOPT_MIMEPOST, mime1);
      curl_easy_setopt(hnd, CURLOPT_USERAGENT, "curl/2000");
      curl_easy_setopt(hnd, CURLOPT_FOLLOWLOCATION, 1L);
      curl_easy_setopt(hnd, CURLOPT_MAXREDIRS, 50L);
      curl_easy_setopt(hnd, CURLOPT_HTTP_VERSION,
                       (long)CURL_HTTP_VERSION_2TLS);
      curl_easy_setopt(hnd, CURLOPT_VERBOSE, 1L);
      curl_easy_setopt(hnd, CURLOPT_FTP_SKIP_PASV_IP, 1L);
      curl_easy_setopt(hnd, CURLOPT_TCP_KEEPALIVE, 1L);
      res = curl_easy_perform(hnd);
    }
  }

  curl_easy_cleanup(hnd);
  curl_mime_free(mime1);
  curl_global_cleanup();
  return (int)res;
}
