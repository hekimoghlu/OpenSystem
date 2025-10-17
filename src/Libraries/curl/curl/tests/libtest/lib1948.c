/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 27, 2023.
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

typedef struct
{
  char *buf;
  size_t len;
} put_buffer;

static size_t put_callback(char *ptr, size_t size, size_t nmemb, void *stream)
{
  put_buffer *putdata = (put_buffer *)stream;
  size_t totalsize = size * nmemb;
  size_t tocopy = (putdata->len < totalsize) ? putdata->len : totalsize;
  memcpy(ptr, putdata->buf, tocopy);
  putdata->len -= tocopy;
  putdata->buf += tocopy;
  return tocopy;
}

int test(char *URL)
{
  CURL *curl;
  CURLcode res = CURLE_OK;
  const char *testput = "This is test PUT data\n";
  put_buffer pbuf;

  curl_global_init(CURL_GLOBAL_DEFAULT);

  easy_init(curl);

  /* PUT */
  easy_setopt(curl, CURLOPT_UPLOAD, 1L);
  easy_setopt(curl, CURLOPT_HEADER, 1L);
  easy_setopt(curl, CURLOPT_READFUNCTION, put_callback);
  pbuf.buf = (char *)testput;
  pbuf.len = strlen(testput);
  easy_setopt(curl, CURLOPT_READDATA, &pbuf);
  easy_setopt(curl, CURLOPT_INFILESIZE, (long)strlen(testput));
  easy_setopt(curl, CURLOPT_URL, URL);
  res = curl_easy_perform(curl);
  if(res)
    goto test_cleanup;

  /* POST */
  easy_setopt(curl, CURLOPT_POST, 1L);
  easy_setopt(curl, CURLOPT_POSTFIELDS, testput);
  easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(testput));
  res = curl_easy_perform(curl);

test_cleanup:
  curl_easy_cleanup(curl);
  curl_global_cleanup();
  return (int)res;
}
