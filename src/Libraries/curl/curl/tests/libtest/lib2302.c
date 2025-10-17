/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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

#ifdef USE_WEBSOCKETS

struct ws_data {
  CURL *easy;
  char buf[1024*1024];
  size_t blen;
  size_t nwrites;
  int has_meta;
  int meta_flags;
};

static void flush_data(struct ws_data *wd)
{
  size_t i;

  if(!wd->nwrites)
    return;

  for(i = 0; i < wd->blen; ++i)
    printf("%02x ", (unsigned char)wd->buf[i]);

  printf("\n");
  if(wd->has_meta)
    printf("RECFLAGS: %x\n", wd->meta_flags);
  else
    fprintf(stderr, "RECFLAGS: NULL\n");
  wd->blen = 0;
  wd->nwrites = 0;
}

static size_t add_data(struct ws_data *wd, const char *buf, size_t blen,
                       const struct curl_ws_frame *meta)
{
  if((wd->nwrites == 0) ||
     (!!meta != !!wd->has_meta) ||
     (meta && meta->flags != wd->meta_flags)) {
    if(wd->nwrites > 0)
      flush_data(wd);
    wd->has_meta = (meta != NULL);
    wd->meta_flags = meta? meta->flags : 0;
  }

  if(wd->blen + blen > sizeof(wd->buf)) {
    return 0;
  }
  memcpy(wd->buf + wd->blen, buf, blen);
  wd->blen += blen;
  wd->nwrites++;
  return blen;
}


static size_t writecb(char *buffer, size_t size, size_t nitems, void *p)
{
  struct ws_data *ws_data = p;
  size_t incoming = nitems;
  const struct curl_ws_frame *meta;
  (void)size;

  meta = curl_ws_meta(ws_data->easy);
  incoming = add_data(ws_data, buffer, incoming, meta);

  if(nitems != incoming)
    fprintf(stderr, "returns error from callback\n");
  return nitems;
}

int test(char *URL)
{
  CURL *curl;
  CURLcode res = CURLE_OK;
  struct ws_data ws_data;


  global_init(CURL_GLOBAL_ALL);

  curl = curl_easy_init();
  if(curl) {
    memset(&ws_data, 0, sizeof(ws_data));
    ws_data.easy = curl;

    curl_easy_setopt(curl, CURLOPT_URL, URL);
    /* use the callback style */
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "webbie-sox/3");
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writecb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ws_data);
    res = curl_easy_perform(curl);
    fprintf(stderr, "curl_easy_perform() returned %u\n", (int)res);
    /* always cleanup */
    curl_easy_cleanup(curl);
    flush_data(&ws_data);
  }
  curl_global_cleanup();
  return (int)res;
}

#else
NO_SUPPORT_BUILT_IN
#endif
