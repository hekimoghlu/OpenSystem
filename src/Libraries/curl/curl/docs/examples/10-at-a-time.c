/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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
 * Download many files in parallel, in the same thread.
 * </DESC>
 */

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#  include <unistd.h>
#endif
#include <curl/curl.h>

static const char *urls[] = {
  "https://www.microsoft.com",
  "https://opensource.org",
  "https://www.google.com",
  "https://www.yahoo.com",
  "https://www.ibm.com",
  "https://www.mysql.com",
  "https://www.oracle.com",
  "https://www.ripe.net",
  "https://www.iana.org",
  "https://www.amazon.com",
  "https://www.netcraft.com",
  "https://www.heise.de",
  "https://www.chip.de",
  "https://www.ca.com",
  "https://www.cnet.com",
  "https://www.mozilla.org",
  "https://www.cnn.com",
  "https://www.wikipedia.org",
  "https://www.dell.com",
  "https://www.hp.com",
  "https://www.cert.org",
  "https://www.mit.edu",
  "https://www.nist.gov",
  "https://www.ebay.com",
  "https://www.playstation.com",
  "https://www.uefa.com",
  "https://www.ieee.org",
  "https://www.apple.com",
  "https://www.symantec.com",
  "https://www.zdnet.com",
  "https://www.fujitsu.com/global/",
  "https://www.supermicro.com",
  "https://www.hotmail.com",
  "https://www.ietf.org",
  "https://www.bbc.co.uk",
  "https://news.google.com",
  "https://www.foxnews.com",
  "https://www.msn.com",
  "https://www.wired.com",
  "https://www.sky.com",
  "https://www.usatoday.com",
  "https://www.cbs.com",
  "https://www.nbc.com/",
  "https://slashdot.org",
  "https://www.informationweek.com",
  "https://apache.org",
  "https://www.un.org",
};

#define MAX_PARALLEL 10 /* number of simultaneous transfers */
#define NUM_URLS sizeof(urls)/sizeof(char *)

static size_t write_cb(char *data, size_t n, size_t l, void *userp)
{
  /* take care of the data here, ignored in this example */
  (void)data;
  (void)userp;
  return n*l;
}

static void add_transfer(CURLM *cm, unsigned int i, int *left)
{
  CURL *eh = curl_easy_init();
  curl_easy_setopt(eh, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(eh, CURLOPT_URL, urls[i]);
  curl_easy_setopt(eh, CURLOPT_PRIVATE, urls[i]);
  curl_multi_add_handle(cm, eh);
  (*left)++;
}

int main(void)
{
  CURLM *cm;
  CURLMsg *msg;
  unsigned int transfers = 0;
  int msgs_left = -1;
  int left = 0;

  curl_global_init(CURL_GLOBAL_ALL);
  cm = curl_multi_init();

  /* Limit the amount of simultaneous connections curl should allow: */
  curl_multi_setopt(cm, CURLMOPT_MAXCONNECTS, (long)MAX_PARALLEL);

  for(transfers = 0; transfers < MAX_PARALLEL && transfers < NUM_URLS;
      transfers++)
    add_transfer(cm, transfers, &left);

  do {
    int still_alive = 1;
    curl_multi_perform(cm, &still_alive);

    while((msg = curl_multi_info_read(cm, &msgs_left))) {
      if(msg->msg == CURLMSG_DONE) {
        char *url;
        CURL *e = msg->easy_handle;
        curl_easy_getinfo(msg->easy_handle, CURLINFO_PRIVATE, &url);
        fprintf(stderr, "R: %d - %s <%s>\n",
                msg->data.result, curl_easy_strerror(msg->data.result), url);
        curl_multi_remove_handle(cm, e);
        curl_easy_cleanup(e);
        left--;
      }
      else {
        fprintf(stderr, "E: CURLMsg (%d)\n", msg->msg);
      }
      if(transfers < NUM_URLS)
        add_transfer(cm, transfers++, &left);
    }
    if(left)
      curl_multi_wait(cm, NULL, 0, 1000, NULL);

  } while(left);

  curl_multi_cleanup(cm);
  curl_global_cleanup();

  return EXIT_SUCCESS;
}

