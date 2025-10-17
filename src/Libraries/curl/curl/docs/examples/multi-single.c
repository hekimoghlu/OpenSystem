/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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
 * using the multi interface to do a single download
 * </DESC>
 */

#include <stdio.h>
#include <string.h>

/* somewhat unix-specific */
#include <sys/time.h>
#include <unistd.h>

/* curl stuff */
#include <curl/curl.h>

/*
 * Simply download an HTTP file.
 */
int main(void)
{
  CURL *http_handle;
  CURLM *multi_handle;
  int still_running = 1; /* keep number of running handles */

  curl_global_init(CURL_GLOBAL_DEFAULT);

  http_handle = curl_easy_init();

  /* set the options (I left out a few, you get the point anyway) */
  curl_easy_setopt(http_handle, CURLOPT_URL, "https://www.example.com/");

  /* init a multi stack */
  multi_handle = curl_multi_init();

  /* add the individual transfers */
  curl_multi_add_handle(multi_handle, http_handle);

  do {
    CURLMcode mc = curl_multi_perform(multi_handle, &still_running);

    if(!mc)
      /* wait for activity, timeout or "nothing" */
      mc = curl_multi_poll(multi_handle, NULL, 0, 1000, NULL);

    if(mc) {
      fprintf(stderr, "curl_multi_poll() failed, code %d.\n", (int)mc);
      break;
    }

  } while(still_running);

  curl_multi_remove_handle(multi_handle, http_handle);

  curl_easy_cleanup(http_handle);

  curl_multi_cleanup(multi_handle);

  curl_global_cleanup();

  return 0;
}
