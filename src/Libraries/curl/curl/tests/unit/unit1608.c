/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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
#include "curlcheck.h"

#include "hostip.h"

#ifndef CURL_DISABLE_SHUFFLE_DNS

CURLcode Curl_shuffle_addr(struct Curl_easy *data,
                           struct Curl_addrinfo **addr);

#define NUM_ADDRS 8
static struct Curl_addrinfo addrs[NUM_ADDRS];

static CURLcode unit_setup(void)
{
  int i;
  for(i = 0; i < NUM_ADDRS - 1; i++) {
    addrs[i].ai_next = &addrs[i + 1];
  }

  return CURLE_OK;
}

static void unit_stop(void)
{
  curl_global_cleanup();
}

UNITTEST_START

  int i;
  CURLcode code;
  struct Curl_addrinfo *addrhead = addrs;

  struct Curl_easy *easy = curl_easy_init();
  abort_unless(easy, "out of memory");

  code = curl_easy_setopt(easy, CURLOPT_DNS_SHUFFLE_ADDRESSES, 1L);
  abort_unless(code == CURLE_OK, "curl_easy_setopt failed");

  /* Shuffle repeatedly and make sure that the list changes */
  for(i = 0; i < 10; i++) {
    if(CURLE_OK != Curl_shuffle_addr(easy, &addrhead))
      break;
    if(addrhead != addrs)
      break;
  }

  curl_easy_cleanup(easy);
  curl_global_cleanup();

  abort_unless(addrhead != addrs, "addresses are not being reordered");

UNITTEST_STOP

#else
static CURLcode unit_setup(void)
{
  return CURLE_OK;
}
static void unit_stop(void)
{
}
UNITTEST_START
UNITTEST_STOP

#endif
