/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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

static char buffer[17000]; /* more than 16K */

int test(char *URL)
{
  CURL *curl = NULL;
  CURLcode res = CURLE_OK;
  curl_mime *mime = NULL;
  curl_mimepart *part;
  size_t i;

  /* Checks huge binary-encoded mime post. */

  /* Create a buffer with pseudo-binary data. */
  for(i = 0; i < sizeof(buffer); i++)
    if(i % 77 == 76)
      buffer[i] = '\n';
    else
      buffer[i] = (char) (0x41 + i % 26); /* A...Z */

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    res = (CURLcode) TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }

  /* Build mime structure. */
  mime = curl_mime_init(curl);
  if(!mime) {
    fprintf(stderr, "curl_mime_init() failed\n");
    res = (CURLcode) TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }
  part = curl_mime_addpart(mime);
  if(!part) {
    fprintf(stderr, "curl_mime_addpart() failed\n");
    res = (CURLcode) TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }
  res = curl_mime_name(part, "upfile");
  if(res) {
    fprintf(stderr, "curl_mime_name() failed\n");
    goto test_cleanup;
  }
  res = curl_mime_filename(part, "myfile.txt");
  if(res) {
    fprintf(stderr, "curl_mime_filename() failed\n");
    goto test_cleanup;
  }
  res = curl_mime_data(part, buffer, sizeof(buffer));
  if(res) {
    fprintf(stderr, "curl_mime_data() failed\n");
    goto test_cleanup;
  }
  res = curl_mime_encoder(part, "binary");
  if(res) {
    fprintf(stderr, "curl_mime_encoder() failed\n");
    goto test_cleanup;
  }

  /* First set the URL that is about to receive our mime mail. */
  test_setopt(curl, CURLOPT_URL, URL);

  /* Post form */
  test_setopt(curl, CURLOPT_MIMEPOST, mime);

  /* Shorten upload buffer. */
  test_setopt(curl, CURLOPT_UPLOAD_BUFFERSIZE, 16411L);

  /* get verbose debug output please */
  test_setopt(curl, CURLOPT_VERBOSE, 1L);

  /* include headers in the output */
  test_setopt(curl, CURLOPT_HEADER, 1L);

  /* Perform the request, res will get the return code */
  res = curl_easy_perform(curl);

test_cleanup:

  /* always cleanup */
  curl_easy_cleanup(curl);

  /* now cleanup the mime structure */
  curl_mime_free(mime);

  curl_global_cleanup();

  return res;
}
