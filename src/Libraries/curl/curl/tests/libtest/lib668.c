/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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

static char data[]= "dummy";

struct WriteThis {
  char *readptr;
  curl_off_t sizeleft;
};

static size_t read_callback(char *ptr, size_t size, size_t nmemb, void *userp)
{
  struct WriteThis *pooh = (struct WriteThis *)userp;
  size_t len = strlen(pooh->readptr);

  (void) size; /* Always 1.*/

  if(len > nmemb)
    len = nmemb;
  if(len) {
    memcpy(ptr, pooh->readptr, len);
    pooh->readptr += len;
  }
  return len;
}

int test(char *URL)
{
  CURL *easy = NULL;
  curl_mime *mime = NULL;
  curl_mimepart *part;
  CURLcode result;
  int res = TEST_ERR_FAILURE;
  struct WriteThis pooh1, pooh2;

  /*
   * Check early end of part data detection.
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

  /* Prepare the callback structures. */
  pooh1.readptr = data;
  pooh1.sizeleft = (curl_off_t) strlen(data);
  pooh2 = pooh1;

  /* Build the mime tree. */
  mime = curl_mime_init(easy);
  part = curl_mime_addpart(mime);
  curl_mime_name(part, "field1");
  /* Early end of data detection can be done because the data size is known. */
  curl_mime_data_cb(part, (curl_off_t) strlen(data),
                    read_callback, NULL, NULL, &pooh1);
  part = curl_mime_addpart(mime);
  curl_mime_name(part, "field2");
  /* Using an undefined length forces chunked transfer and disables early
     end of data detection for this part. */
  curl_mime_data_cb(part, (curl_off_t) -1, read_callback, NULL, NULL, &pooh2);
  part = curl_mime_addpart(mime);
  curl_mime_name(part, "field3");
  /* Regular file part sources early end of data can be detected because
     the file size is known. In addition, and EOF test is performed. */
  curl_mime_filedata(part, libtest_arg2);

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
