/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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

static const char * const post[]={
  "one",
  "two",
  "three",
  "and a final longer crap: four",
  NULL
};


struct WriteThis {
  int counter;
};

static int progress_callback(void *clientp, double dltotal, double dlnow,
                             double ultotal, double ulnow)
{
  static int prev_ultotal = -1;
  static int prev_ulnow = -1;
  (void)clientp; /* UNUSED */
  (void)dltotal; /* UNUSED */
  (void)dlnow; /* UNUSED */

  /* to avoid depending on timing, which will cause this progress function to
     get called a different number of times depending on circumstances, we
     only log these lines if the numbers are different from the previous
     invoke */
  if((prev_ultotal != (int)ultotal) ||
     (prev_ulnow != (int)ulnow)) {

    FILE *moo = fopen(libtest_arg2, "ab");
    if(moo) {
      fprintf(moo, "Progress callback called with UL %d out of %d\n",
              (int)ulnow, (int)ultotal);
      fclose(moo);
    }
    prev_ulnow = (int) ulnow;
    prev_ultotal = (int) ultotal;
  }
  return 0;
}

static size_t read_callback(char *ptr, size_t size, size_t nmemb, void *userp)
{
  struct WriteThis *pooh = (struct WriteThis *)userp;
  const char *data;

  if(size*nmemb < 1)
    return 0;

  data = post[pooh->counter];

  if(data) {
    size_t len = strlen(data);
    memcpy(ptr, data, len);
    pooh->counter++; /* advance pointer */
    return len;
  }
  return 0;                         /* no more data left to deliver */
}

int test(char *URL)
{
  CURL *curl;
  CURLcode res = CURLE_OK;
  struct curl_slist *slist = NULL;
  struct WriteThis pooh;
  pooh.counter = 0;

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }

  slist = curl_slist_append(slist, "Transfer-Encoding: chunked");
  if(!slist) {
    fprintf(stderr, "curl_slist_append() failed\n");
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }

  /* First set the URL that is about to receive our POST. */
  test_setopt(curl, CURLOPT_URL, URL);

  /* Now specify we want to POST data */
  test_setopt(curl, CURLOPT_POST, 1L);

  /* we want to use our own read function */
  test_setopt(curl, CURLOPT_READFUNCTION, read_callback);

  /* pointer to pass to our read function */
  test_setopt(curl, CURLOPT_READDATA, &pooh);

  /* get verbose debug output please */
  test_setopt(curl, CURLOPT_VERBOSE, 1L);

  /* include headers in the output */
  test_setopt(curl, CURLOPT_HEADER, 1L);

  /* enforce chunked transfer by setting the header */
  test_setopt(curl, CURLOPT_HTTPHEADER, slist);

  test_setopt(curl, CURLOPT_HTTPAUTH, (long)CURLAUTH_DIGEST);
  test_setopt(curl, CURLOPT_USERPWD, "foo:bar");

  /* we want to use our own progress function */
  test_setopt(curl, CURLOPT_NOPROGRESS, 0L);
  CURL_IGNORE_DEPRECATION(
    test_setopt(curl, CURLOPT_PROGRESSFUNCTION, progress_callback);
  )

  /* Perform the request, res will get the return code */
  res = curl_easy_perform(curl);

test_cleanup:

  /* clean up the headers list */
  if(slist)
    curl_slist_free_all(slist);

  /* always cleanup */
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return res;
}
