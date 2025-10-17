/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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

int test(char *URL)
{
  CURLSH *sh = NULL;
  CURL *ch = NULL;
  int unfinished;
  CURLM *cm;

  curl_global_init(CURL_GLOBAL_ALL);

  cm = curl_multi_init();
  if(!cm) {
    curl_global_cleanup();
    return 1;
  }
  sh = curl_share_init();
  if(!sh)
    goto cleanup;

  curl_share_setopt(sh, CURLSHOPT_SHARE, CURL_LOCK_DATA_COOKIE);
  curl_share_setopt(sh, CURLSHOPT_SHARE, CURL_LOCK_DATA_COOKIE);

  ch = curl_easy_init();
  if(!ch)
    goto cleanup;

  curl_easy_setopt(ch, CURLOPT_SHARE, sh);
  curl_easy_setopt(ch, CURLOPT_URL, URL);
  curl_easy_setopt(ch, CURLOPT_COOKIEFILE, libtest_arg2);
  curl_easy_setopt(ch, CURLOPT_COOKIEJAR, libtest_arg2);

  curl_multi_add_handle(cm, ch);

  unfinished = 1;
  while(unfinished) {
    int MAX = 0;
    long max_tout;
    fd_set R, W, E;
    struct timeval timeout;

    FD_ZERO(&R);
    FD_ZERO(&W);
    FD_ZERO(&E);
    curl_multi_perform(cm, &unfinished);

    curl_multi_fdset(cm, &R, &W, &E, &MAX);
    curl_multi_timeout(cm, &max_tout);

    if(max_tout > 0) {
      curlx_mstotv(&timeout, max_tout);
    }
    else {
      timeout.tv_sec = 0;
      timeout.tv_usec = 1000;
    }

    select(MAX + 1, &R, &W, &E, &timeout);
  }

  curl_easy_setopt(ch, CURLOPT_COOKIELIST, "FLUSH");
  curl_easy_setopt(ch, CURLOPT_SHARE, NULL);

  curl_multi_remove_handle(cm, ch);
cleanup:
  curl_easy_cleanup(ch);
  curl_share_cleanup(sh);
  curl_multi_cleanup(cm);
  curl_global_cleanup();

  return 0;
}
