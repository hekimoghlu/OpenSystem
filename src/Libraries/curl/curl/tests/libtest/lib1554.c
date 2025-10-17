/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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

static void my_lock(CURL *handle, curl_lock_data data,
                    curl_lock_access laccess, void *useptr)
{
  (void)handle;
  (void)data;
  (void)laccess;
  (void)useptr;
  printf("-> Mutex lock\n");
}

static void my_unlock(CURL *handle, curl_lock_data data, void *useptr)
{
  (void)handle;
  (void)data;
  (void)useptr;
  printf("<- Mutex unlock\n");
}

/* test function */
int test(char *URL)
{
  CURLcode res = CURLE_OK;
  CURLSH *share = NULL;
  int i;

  global_init(CURL_GLOBAL_ALL);

  share = curl_share_init();
  if(!share) {
    fprintf(stderr, "curl_share_init() failed\n");
    goto test_cleanup;
  }

  curl_share_setopt(share, CURLSHOPT_SHARE, CURL_LOCK_DATA_CONNECT);
  curl_share_setopt(share, CURLSHOPT_LOCKFUNC, my_lock);
  curl_share_setopt(share, CURLSHOPT_UNLOCKFUNC, my_unlock);

  /* Loop the transfer and cleanup the handle properly every lap. This will
     still reuse connections since the pool is in the shared object! */

  for(i = 0; i < 3; i++) {
    CURL *curl = curl_easy_init();
    if(curl) {
      curl_easy_setopt(curl, CURLOPT_URL, URL);

      /* use the share object */
      curl_easy_setopt(curl, CURLOPT_SHARE, share);

      /* Perform the request, res will get the return code */
      res = curl_easy_perform(curl);

      /* always cleanup */
      curl_easy_cleanup(curl);

      /* Check for errors */
      if(res != CURLE_OK) {
        fprintf(stderr, "curl_easy_perform() failed: %s\n",
                curl_easy_strerror(res));
        goto test_cleanup;
      }
    }
  }

test_cleanup:
  curl_share_cleanup(share);
  curl_global_cleanup();

  return (int)res;
}
