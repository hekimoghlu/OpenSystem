/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
  "dummy\n";

struct WriteThis {
  char *readptr;
  curl_off_t sizeleft;
  int freecount;
};

static void free_callback(void *userp)
{
  struct WriteThis *pooh = (struct WriteThis *) userp;

  pooh->freecount++;
}

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
  CURL *easy2 = NULL;
  curl_mime *mime = NULL;
  curl_mimepart *part;
  struct curl_slist *hdrs = NULL;
  CURLcode result;
  int res = TEST_ERR_FAILURE;
  struct WriteThis pooh;

  /*
   * Check proper copy/release of mime post data bound to a duplicated
   * easy handle.
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
  pooh.freecount = 0;

  /* Build the mime tree. */
  mime = curl_mime_init(easy);
  part = curl_mime_addpart(mime);
  curl_mime_data(part, "hello", CURL_ZERO_TERMINATED);
  curl_mime_name(part, "greeting");
  curl_mime_type(part, "application/X-Greeting");
  curl_mime_encoder(part, "base64");
  hdrs = curl_slist_append(hdrs, "X-Test-Number: 654");
  curl_mime_headers(part, hdrs, TRUE);
  part = curl_mime_addpart(mime);
  curl_mime_filedata(part, libtest_arg2);
  part = curl_mime_addpart(mime);
  curl_mime_data_cb(part, (curl_off_t) -1, read_callback, NULL, free_callback,
                    &pooh);

  /* Bind mime data to its easy handle. */
  test_setopt(easy, CURLOPT_MIMEPOST, mime);

  /* Duplicate the handle. */
  easy2 = curl_easy_duphandle(easy);
  if(!easy2) {
    fprintf(stderr, "curl_easy_duphandle() failed\n");
    res = TEST_ERR_FAILURE;
    goto test_cleanup;
  }

  /* Now free the mime structure: it should unbind it from the first
     easy handle. */
  curl_mime_free(mime);
  mime = NULL;  /* Already cleaned up. */

  /* Perform on the first handle: should not send any data. */
  result = curl_easy_perform(easy);
  if(result) {
    fprintf(stderr, "curl_easy_perform(original) failed\n");
    res = (int) result;
    goto test_cleanup;
  }

  /* Perform on the second handle: if the bound mime structure has not been
     duplicated properly, it should cause a valgrind error. */
  result = curl_easy_perform(easy2);
  if(result) {
    fprintf(stderr, "curl_easy_perform(duplicated) failed\n");
    res = (int) result;
    goto test_cleanup;
  }

  /* Free the duplicated handle: it should call free_callback again.
     If the mime copy was bad or not automatically released, valgrind
     will signal it. */
  curl_easy_cleanup(easy2);
  easy2 = NULL;  /* Already cleaned up. */

  if(pooh.freecount != 2) {
    fprintf(stderr, "free_callback() called %d times instead of 2\n",
            pooh.freecount);
    res = TEST_ERR_FAILURE;
    goto test_cleanup;
  }

test_cleanup:
  curl_easy_cleanup(easy);
  curl_easy_cleanup(easy2);
  curl_mime_free(mime);
  curl_global_cleanup();
  return res;
}
