/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 11, 2024.
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
 * Preload domains to HSTS
 * </DESC>
 */
#include <stdio.h>
#include <string.h>
#include <curl/curl.h>

struct entry {
  const char *name;
  const char *exp;
};

static const struct entry preload_hosts[] = {
  { "example.com", "20370320 01:02:03" },
  { "curl.se",     "20370320 03:02:01" },
  { NULL, NULL } /* end of list marker */
};

struct state {
  int index;
};

/* "read" is from the point of the library, it wants data from us. One domain
   entry per invoke. */
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
    e->includeSubDomains = 0;
    strcpy(e->expire, expire);
    fprintf(stderr, "HSTS preload '%s' until '%s'\n", host, expire);
  }
  else
    return CURLSTS_DONE;
  return CURLSTS_OK;
}

static CURLSTScode hstswrite(CURL *easy, struct curl_hstsentry *e,
                             struct curl_index *i, void *userp)
{
  (void)easy;
  (void)userp; /* we have no custom input */
  printf("[%u/%u] %s %s\n", (unsigned int)i->index, (unsigned int)i->total,
         e->name, e->expire);
  return CURLSTS_OK;
}

int main(void)
{
  CURL *curl;
  CURLcode res;

  curl = curl_easy_init();
  if(curl) {
    struct state st = {0};

    /* enable HSTS for this handle */
    curl_easy_setopt(curl, CURLOPT_HSTS_CTRL, (long)CURLHSTS_ENABLE);

    /* function to call at first to populate the cache before the transfer */
    curl_easy_setopt(curl, CURLOPT_HSTSREADFUNCTION, hstsread);
    curl_easy_setopt(curl, CURLOPT_HSTSREADDATA, &st);

    /* function to call after transfer to store the new state of the HSTS
       cache */
    curl_easy_setopt(curl, CURLOPT_HSTSWRITEFUNCTION, hstswrite);
    curl_easy_setopt(curl, CURLOPT_HSTSWRITEDATA, NULL);

    /* use the domain with HTTP but due to the preload, it should do the
       transfer using HTTPS */
    curl_easy_setopt(curl, CURLOPT_URL, "http://curl.se");

    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

    /* Perform the request, res gets the return code */
    res = curl_easy_perform(curl);
    /* Check for errors */
    if(res != CURLE_OK)
      fprintf(stderr, "curl_easy_perform() failed: %s\n",
              curl_easy_strerror(res));

    /* always cleanup */
    curl_easy_cleanup(curl);
  }
  return 0;
}
