/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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
 * A basic application source code using the multi interface doing two
 * transfers in parallel.
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
 * Download an HTTP file and upload an FTP file simultaneously.
 */

#define HANDLECOUNT 2   /* Number of simultaneous transfers */
#define HTTP_HANDLE 0   /* Index for the HTTP transfer */
#define FTP_HANDLE 1    /* Index for the FTP transfer */

int main(void)
{
  CURL *handles[HANDLECOUNT];
  CURLM *multi_handle;

  int still_running = 1; /* keep number of running handles */
  int i;

  CURLMsg *msg; /* for picking up messages with the transfer status */
  int msgs_left; /* how many messages are left */

  /* Allocate one CURL handle per transfer */
  for(i = 0; i<HANDLECOUNT; i++)
    handles[i] = curl_easy_init();

  /* set the options (I left out a few, you get the point anyway) */
  curl_easy_setopt(handles[HTTP_HANDLE], CURLOPT_URL, "https://example.com");

  curl_easy_setopt(handles[FTP_HANDLE], CURLOPT_URL, "ftp://example.com");
  curl_easy_setopt(handles[FTP_HANDLE], CURLOPT_UPLOAD, 1L);

  /* init a multi stack */
  multi_handle = curl_multi_init();

  /* add the individual transfers */
  for(i = 0; i<HANDLECOUNT; i++)
    curl_multi_add_handle(multi_handle, handles[i]);

  while(still_running) {
    CURLMcode mc = curl_multi_perform(multi_handle, &still_running);

    if(still_running)
      /* wait for activity, timeout or "nothing" */
      mc = curl_multi_poll(multi_handle, NULL, 0, 1000, NULL);

    if(mc)
      break;
  }
  /* See how the transfers went */
  while((msg = curl_multi_info_read(multi_handle, &msgs_left))) {
    if(msg->msg == CURLMSG_DONE) {
      int idx;

      /* Find out which handle this message is about */
      for(idx = 0; idx<HANDLECOUNT; idx++) {
        int found = (msg->easy_handle == handles[idx]);
        if(found)
          break;
      }

      switch(idx) {
      case HTTP_HANDLE:
        printf("HTTP transfer completed with status %d\n", msg->data.result);
        break;
      case FTP_HANDLE:
        printf("FTP transfer completed with status %d\n", msg->data.result);
        break;
      }
    }
  }

  /* remove the transfers and cleanup the handles */
  for(i = 0; i<HANDLECOUNT; i++) {
    curl_multi_remove_handle(multi_handle, handles[i]);
    curl_easy_cleanup(handles[i]);
  }

  curl_multi_cleanup(multi_handle);

  return 0;
}
