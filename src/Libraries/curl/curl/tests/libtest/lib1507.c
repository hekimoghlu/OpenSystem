/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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

#include "testutil.h"
#include "timediff.h"
#include "warnless.h"
#include "memdebug.h"

/*
 * This is the list of basic details you need to tweak to get things right.
 */
#define USERNAME "user@example.com"
#define PASSWORD "123qwerty"
#define RECIPIENT "<1507-recipient@example.com>"
#define MAILFROM "<1507-realuser@example.com>"

#define MULTI_PERFORM_HANG_TIMEOUT 60 * 1000

static size_t read_callback(char *ptr, size_t size, size_t nmemb, void *userp)
{
  (void)ptr;
  (void)size;
  (void)nmemb;
  (void)userp;
  return CURL_READFUNC_ABORT;
}

int test(char *URL)
{
   int res = 0;
   CURL *curl = NULL;
   CURLM *mcurl = NULL;
   int still_running = 1;
   struct timeval mp_start;
   struct curl_slist *rcpt_list = NULL;

   curl_global_init(CURL_GLOBAL_DEFAULT);

   easy_init(curl);

   multi_init(mcurl);

   rcpt_list = curl_slist_append(rcpt_list, RECIPIENT);
   /* more addresses can be added here
      rcpt_list = curl_slist_append(rcpt_list, "<others@example.com>");
   */

   curl_easy_setopt(curl, CURLOPT_URL, URL);
#if 0
   curl_easy_setopt(curl, CURLOPT_USERNAME, USERNAME);
   curl_easy_setopt(curl, CURLOPT_PASSWORD, PASSWORD);
#endif
   curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
   curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_callback);
   curl_easy_setopt(curl, CURLOPT_MAIL_FROM, MAILFROM);
   curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, rcpt_list);
   curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
   multi_add_handle(mcurl, curl);

   mp_start = tutil_tvnow();

  /* we start some action by calling perform right away */
  curl_multi_perform(mcurl, &still_running);

  while(still_running) {
    struct timeval timeout;
    int rc; /* select() return code */

    fd_set fdread;
    fd_set fdwrite;
    fd_set fdexcep;
    int maxfd = -1;

    long curl_timeo = -1;

    FD_ZERO(&fdread);
    FD_ZERO(&fdwrite);
    FD_ZERO(&fdexcep);

    /* set a suitable timeout to play around with */
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;

    curl_multi_timeout(mcurl, &curl_timeo);
    if(curl_timeo >= 0) {
      curlx_mstotv(&timeout, curl_timeo);
      if(timeout.tv_sec > 1) {
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
      }
    }

    /* get file descriptors from the transfers */
    curl_multi_fdset(mcurl, &fdread, &fdwrite, &fdexcep, &maxfd);

    /* In a real-world program you OF COURSE check the return code of the
       function calls.  On success, the value of maxfd is guaranteed to be
       greater or equal than -1.  We call select(maxfd + 1, ...), specially in
       case of (maxfd == -1), we call select(0, ...), which is basically equal
       to sleep. */

    rc = select(maxfd + 1, &fdread, &fdwrite, &fdexcep, &timeout);

    if(tutil_tvdiff(tutil_tvnow(), mp_start) > MULTI_PERFORM_HANG_TIMEOUT) {
      fprintf(stderr, "ABORTING TEST, since it seems "
              "that it would have run forever.\n");
      break;
    }

    switch(rc) {
    case -1:
      /* select error */
      break;
    case 0: /* timeout */
    default: /* action */
      curl_multi_perform(mcurl, &still_running);
      break;
    }
  }

test_cleanup:

  curl_slist_free_all(rcpt_list);
  curl_multi_remove_handle(mcurl, curl);
  curl_multi_cleanup(mcurl);
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return res;
}
