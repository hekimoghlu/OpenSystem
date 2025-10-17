/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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
#include "warnless.h"
#include "memdebug.h"

struct entry {
  const char *name;
  const char *exp;
};

static const struct entry preload_hosts[] = {
#if (SIZEOF_TIME_T < 5)
  { "1.example.com", "20370320 01:02:03" },
  { "2.example.com", "20370320 03:02:01" },
  { "3.example.com", "20370319 01:02:03" },
#else
  { "1.example.com", "25250320 01:02:03" },
  { "2.example.com", "25250320 03:02:01" },
  { "3.example.com", "25250319 01:02:03" },
#endif
  { "4.example.com", "" },
  { NULL, NULL } /* end of list marker */
};

struct state {
  int index;
};

/* "read" is from the point of the library, it wants data from us */
static CURLSTScode hstsread(CURL *easy, struct curl_hstsentry *e,
                            void *userp)
{
  const char *host;
  const char *expire;
  struct state *s = (struct state *)userp;
  (void)easy;
  host = preload_hosts[s->index].name;
  expire = preload_hosts[s->index++].exp;

  if(host && (strlen(host) < e->namelen)) {
    strcpy(e->name, host);
    e->includeSubDomains = FALSE;
    strcpy(e->expire, expire);
    fprintf(stderr, "add '%s'\n", host);
  }
  else
    return CURLSTS_DONE;
  return CURLSTS_OK;
}

/* verify error from callback */
static CURLSTScode hstsreadfail(CURL *easy, struct curl_hstsentry *e,
                                void *userp)
{
  (void)easy;
  (void)e;
  (void)userp;
  return CURLSTS_FAIL;
}

/* check that we get the hosts back in the save */
static CURLSTScode hstswrite(CURL *easy, struct curl_hstsentry *e,
                             struct curl_index *i, void *userp)
{
  (void)easy;
  (void)userp;
  printf("[%zu/%zu] %s %s\n", i->index, i->total, e->name, e->expire);
  return CURLSTS_OK;
}

/*
 * Read/write HSTS cache entries via callback.
 */

int test(char *URL)
{
  CURLcode res = CURLE_OK;
  CURL *hnd;
  struct state st = {0};

  global_init(CURL_GLOBAL_ALL);

  easy_init(hnd);
  easy_setopt(hnd, CURLOPT_URL, URL);
  easy_setopt(hnd, CURLOPT_HSTSREADFUNCTION, hstsread);
  easy_setopt(hnd, CURLOPT_HSTSREADDATA, &st);
  easy_setopt(hnd, CURLOPT_HSTSWRITEFUNCTION, hstswrite);
  easy_setopt(hnd, CURLOPT_HSTSWRITEDATA, &st);
  easy_setopt(hnd, CURLOPT_HSTS_CTRL, CURLHSTS_ENABLE);
  res = curl_easy_perform(hnd);
  curl_easy_cleanup(hnd);
  hnd = NULL;
  printf("First request returned %d\n", (int)res);
  res = CURLE_OK;

  easy_init(hnd);
  easy_setopt(hnd, CURLOPT_URL, URL);
  easy_setopt(hnd, CURLOPT_HSTSREADFUNCTION, hstsreadfail);
  easy_setopt(hnd, CURLOPT_HSTSREADDATA, &st);
  easy_setopt(hnd, CURLOPT_HSTSWRITEFUNCTION, hstswrite);
  easy_setopt(hnd, CURLOPT_HSTSWRITEDATA, &st);
  easy_setopt(hnd, CURLOPT_HSTS_CTRL, CURLHSTS_ENABLE);
  res = curl_easy_perform(hnd);
  curl_easy_cleanup(hnd);
  hnd = NULL;
  printf("Second request returned %d\n", (int)res);

test_cleanup:
  curl_easy_cleanup(hnd);
  curl_global_cleanup();
  return (int)res;
}
