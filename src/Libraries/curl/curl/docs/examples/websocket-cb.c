/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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
 * WebSocket download-only using write callback
 * </DESC>
 */
#include <stdio.h>
#include <curl/curl.h>

static size_t writecb(char *b, size_t size, size_t nitems, void *p)
{
  CURL *easy = p;
  size_t i;
  const struct curl_ws_frame *frame = curl_ws_meta(easy);
  fprintf(stderr, "Type: %s\n", frame->flags & CURLWS_BINARY ?
          "binary" : "text");
  fprintf(stderr, "Bytes: %u", (unsigned int)(nitems * size));
  for(i = 0; i < nitems; i++)
    fprintf(stderr, "%02x ", (unsigned char)b[i]);
  return nitems;
}

int main(void)
{
  CURL *curl;
  CURLcode res;

  curl = curl_easy_init();
  if(curl) {
    curl_easy_setopt(curl, CURLOPT_URL, "wss://example.com");

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writecb);
    /* pass the easy handle to the callback */
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, curl);

    /* Perform the request, res gets the return code */
    res = curl_easy_perform(curl);
    /* Check for errors */
    if(res != CURLE_OK)
      fprintf(stderr, "curl_easy_perform() failed: %s\n",
              curl_easy_strerror(res));

    /* always cleanup */
    curl_easy_cleanup(curl);
  }
  return 0;
}
