/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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
 * Use the progress callbacks, old and/or new one depending on available
 * libcurl version.
 * </DESC>
 */
#include <stdio.h>
#include <curl/curl.h>

#define MINIMAL_PROGRESS_FUNCTIONALITY_INTERVAL     3000000
#define STOP_DOWNLOAD_AFTER_THIS_MANY_BYTES         6000

struct myprogress {
  curl_off_t lastruntime; /* type depends on version, see above */
  CURL *curl;
};

/* this is how the CURLOPT_XFERINFOFUNCTION callback works */
static int xferinfo(void *p,
                    curl_off_t dltotal, curl_off_t dlnow,
                    curl_off_t ultotal, curl_off_t ulnow)
{
  struct myprogress *myp = (struct myprogress *)p;
  CURL *curl = myp->curl;
  curl_off_t curtime = 0;

  curl_easy_getinfo(curl, CURLINFO_TOTAL_TIME_T, &curtime);

  /* under certain circumstances it may be desirable for certain functionality
     to only run every N seconds, in order to do this the transaction time can
     be used */
  if((curtime - myp->lastruntime) >= MINIMAL_PROGRESS_FUNCTIONALITY_INTERVAL) {
    myp->lastruntime = curtime;
    fprintf(stderr, "TOTAL TIME: %lu.%06lu\r\n",
            (unsigned long)(curtime / 1000000),
            (unsigned long)(curtime % 1000000));
  }

  fprintf(stderr, "UP: %lu of %lu  DOWN: %lu of %lu\r\n",
          (unsigned long)ulnow, (unsigned long)ultotal,
          (unsigned long)dlnow, (unsigned long)dltotal);

  if(dlnow > STOP_DOWNLOAD_AFTER_THIS_MANY_BYTES)
    return 1;
  return 0;
}

int main(void)
{
  CURL *curl;
  CURLcode res = CURLE_OK;
  struct myprogress prog;

  curl = curl_easy_init();
  if(curl) {
    prog.lastruntime = 0;
    prog.curl = curl;

    curl_easy_setopt(curl, CURLOPT_URL, "https://example.com/");

    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, xferinfo);
    /* pass the struct pointer into the xferinfo function */
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &prog);

    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    res = curl_easy_perform(curl);

    if(res != CURLE_OK)
      fprintf(stderr, "%s\n", curl_easy_strerror(res));

    /* always cleanup */
    curl_easy_cleanup(curl);
  }
  return (int)res;
}
