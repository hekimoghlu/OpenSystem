/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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
/*
 * Make sure libcurl does not send a `Content-Length: -1` header when HTTP POST
 * size is unknown.
 */

#include "test.h"

#include "memdebug.h"

static char data[]="dummy";

struct WriteThis {
  char *readptr;
  size_t sizeleft;
};

static size_t read_callback(char *ptr, size_t size, size_t nmemb, void *userp)
{
  struct WriteThis *pooh = (struct WriteThis *)userp;

  if(size*nmemb < 1)
    return 0;

  if(pooh->sizeleft) {
    *ptr = pooh->readptr[0]; /* copy one single byte */
    pooh->readptr++;                 /* advance pointer */
    pooh->sizeleft--;                /* less data left */
    return 1;                        /* we return 1 byte at a time! */
  }

  return 0;                         /* no more data left to deliver */
}

int test(char *URL)
{
  CURL *curl;
  CURLcode result = CURLE_OK;
  int res = 0;
  struct WriteThis pooh = { data, sizeof(data)-1 };

  global_init(CURL_GLOBAL_ALL);

  easy_init(curl);

  easy_setopt(curl, CURLOPT_URL, URL);
  easy_setopt(curl, CURLOPT_POST, 1L);
  /* Purposely omit to set CURLOPT_POSTFIELDSIZE */
  easy_setopt(curl, CURLOPT_READFUNCTION, read_callback);
  easy_setopt(curl, CURLOPT_READDATA, &pooh);
#ifdef LIB1539
  /* speak HTTP 1.0 - no chunked! */
  easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_0);
#endif

  result = curl_easy_perform(curl);

test_cleanup:

  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return (int)result;
}

