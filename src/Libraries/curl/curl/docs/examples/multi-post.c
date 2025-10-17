/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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
 * using the multi interface to do a multipart formpost without blocking
 * </DESC>
 */

#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <curl/curl.h>

int main(void)
{
  CURL *curl;

  CURLM *multi_handle;
  int still_running = 0;

  curl_mime *form = NULL;
  curl_mimepart *field = NULL;
  struct curl_slist *headerlist = NULL;
  static const char buf[] = "Expect:";

  curl = curl_easy_init();
  multi_handle = curl_multi_init();

  if(curl && multi_handle) {
    /* Create the form */
    form = curl_mime_init(curl);

    /* Fill in the file upload field */
    field = curl_mime_addpart(form);
    curl_mime_name(field, "sendfile");
    curl_mime_filedata(field, "multi-post.c");

    /* Fill in the filename field */
    field = curl_mime_addpart(form);
    curl_mime_name(field, "filename");
    curl_mime_data(field, "multi-post.c", CURL_ZERO_TERMINATED);

    /* Fill in the submit field too, even if this is rarely needed */
    field = curl_mime_addpart(form);
    curl_mime_name(field, "submit");
    curl_mime_data(field, "send", CURL_ZERO_TERMINATED);

    /* initialize custom header list (stating that Expect: 100-continue is not
       wanted */
    headerlist = curl_slist_append(headerlist, buf);

    /* what URL that receives this POST */
    curl_easy_setopt(curl, CURLOPT_URL, "https://www.example.com/upload.cgi");
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerlist);
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

    curl_multi_add_handle(multi_handle, curl);

    do {
      CURLMcode mc = curl_multi_perform(multi_handle, &still_running);

      if(still_running)
        /* wait for activity, timeout or "nothing" */
        mc = curl_multi_poll(multi_handle, NULL, 0, 1000, NULL);

      if(mc)
        break;
    } while(still_running);

    curl_multi_cleanup(multi_handle);

    /* always cleanup */
    curl_easy_cleanup(curl);

    /* then cleanup the form */
    curl_mime_free(form);

    /* free slist */
    curl_slist_free_all(headerlist);
  }
  return 0;
}

