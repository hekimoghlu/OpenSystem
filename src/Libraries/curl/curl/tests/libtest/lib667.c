/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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

#include "memdebug.h"

static char data[]=
  "dummy";

struct WriteThis {
  char *readptr;
  curl_off_t sizeleft;
};

static size_t read_callback(char *ptr, size_t size, size_t nmemb, void *userp)
{
  struct WriteThis *pooh = (struct WriteThis *)userp;
  int eof = !*pooh->readptr;

  if(size*nmemb < 1)
    return 0;

  eof = pooh->sizeleft <= 0;
  if(!eof)
    pooh->sizeleft--;

  if(!eof) {
    *ptr = *pooh->readptr;           /* copy one single byte */
    pooh->readptr++;                 /* advance pointer */
    return 1;                        /* we return 1 byte at a time! */
  }

  return 0;                         /* no more data left to deliver */
}

int test(char *URL)
{
  CURL *easy = NULL;
  curl_mime *mime = NULL;
  curl_mimepart *part;
  CURLcode result;
  int res = TEST_ERR_FAILURE;
  struct WriteThis pooh;

  /*
   * Check proper handling of mime encoder feature when the part read callback
   * delivers data bytes one at a time. Use chunked encoding for accurate test.
   */

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  easy = curl_easy_init();

  /* First set the URL that is about to receive our POST. */
  test_setopt(easy, CURLOPT_URL, URL);

  /* get verbose debug output please */
  test_setopt(easy, CURLOPT_VERBOSE, 1L);

  /* include headers in the output */
  test_setopt(easy, CURLOPT_HEADER, 1L);

  /* Prepare the callback structure. */
  pooh.readptr = data;
  pooh.sizeleft = (curl_off_t) strlen(data);

  /* Build the mime tree. */
  mime = curl_mime_init(easy);
  part = curl_mime_addpart(mime);
  curl_mime_name(part, "field");
  curl_mime_encoder(part, "base64");
  /* Using an undefined length forces chunked transfer. */
  curl_mime_data_cb(part, (curl_off_t) -1, read_callback, NULL, NULL, &pooh);

  /* Bind mime data to its easy handle. */
  test_setopt(easy, CURLOPT_MIMEPOST, mime);

  /* Send data. */
  result = curl_easy_perform(easy);
  if(result) {
    fprintf(stderr, "curl_easy_perform() failed\n");
    res = (int) result;
  }

test_cleanup:
  curl_easy_cleanup(easy);
  curl_mime_free(mime);
  curl_global_cleanup();
  return res;
}
