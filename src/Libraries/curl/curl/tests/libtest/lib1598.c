/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 20, 2025.
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
 * This unit test PUT http data over proxy. Proxy header will be different
 * from server http header
 */

#include "test.h"
#include <stdio.h>
#include "memdebug.h"

/*
 * carefully not leak memory on OOM
 */
static int trailers_callback(struct curl_slist **list, void *userdata)
{
  struct curl_slist *nlist = NULL;
  struct curl_slist *nlist2 = NULL;
  (void)userdata;
  nlist = curl_slist_append(*list, "my-super-awesome-trailer: trail1");
  if(nlist)
    nlist2 = curl_slist_append(nlist, "my-other-awesome-trailer: trail2");
  if(nlist2) {
    *list = nlist2;
    return CURL_TRAILERFUNC_OK;
  }
  else {
    curl_slist_free_all(nlist);
    return CURL_TRAILERFUNC_ABORT;
  }
}

static const char *post_data = "xxx=yyy&aaa=bbbbb";

int test(char *URL)
{
  CURL *curl = NULL;
  CURLcode res = CURLE_FAILED_INIT;
  /* http and proxy header list */
  struct curl_slist *hhl = NULL, *list;

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

  hhl = curl_slist_append(hhl, "Trailer: my-super-awesome-trailer,"
                               " my-other-awesome-trailer");
  if(!hhl)
    goto test_cleanup;
  if(hhl) {
    list = curl_slist_append(hhl, "Transfer-Encoding: chunked");
    if(!list)
      goto test_cleanup;
    hhl = list;
  }

  test_setopt(curl, CURLOPT_URL, URL);
  test_setopt(curl, CURLOPT_HTTPHEADER, hhl);
  test_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(post_data));
  test_setopt(curl, CURLOPT_POSTFIELDS, post_data);
  test_setopt(curl, CURLOPT_TRAILERFUNCTION, trailers_callback);
  test_setopt(curl, CURLOPT_TRAILERDATA, NULL);
  test_setopt(curl, CURLOPT_VERBOSE, 1L);

  res = curl_easy_perform(curl);

test_cleanup:

  curl_easy_cleanup(curl);

  curl_slist_free_all(hhl);

  curl_global_cleanup();

  return (int)res;
}
