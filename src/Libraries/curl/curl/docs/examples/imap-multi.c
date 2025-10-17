/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
 * Get IMAP email with the multi interface
 * </DESC>
 */

#include <stdio.h>
#include <string.h>
#include <curl/curl.h>

/* This is a simple example showing how to fetch mail using libcurl's IMAP
 * capabilities. It builds on the imap-fetch.c example to demonstrate how to
 * use libcurl's multi interface.
 */

int main(void)
{
  CURL *curl;
  CURLM *mcurl;
  int still_running = 1;

  curl_global_init(CURL_GLOBAL_DEFAULT);

  curl = curl_easy_init();
  if(!curl)
    return 1;

  mcurl = curl_multi_init();
  if(!mcurl)
    return 2;

  /* Set username and password */
  curl_easy_setopt(curl, CURLOPT_USERNAME, "user");
  curl_easy_setopt(curl, CURLOPT_PASSWORD, "secret");

  /* This fetches message 1 from the user's inbox */
  curl_easy_setopt(curl, CURLOPT_URL, "imap://imap.example.com/INBOX/;UID=1");

  /* Tell the multi stack about our easy handle */
  curl_multi_add_handle(mcurl, curl);

  do {
    CURLMcode mc = curl_multi_perform(mcurl, &still_running);

    if(still_running)
      /* wait for activity, timeout or "nothing" */
      mc = curl_multi_poll(mcurl, NULL, 0, 1000, NULL);

    if(mc)
      break;
  } while(still_running);

  /* Always cleanup */
  curl_multi_remove_handle(mcurl, curl);
  curl_multi_cleanup(mcurl);
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return 0;
}
