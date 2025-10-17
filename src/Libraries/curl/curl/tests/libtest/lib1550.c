/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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

#include "memdebug.h"

#include <curl/multi.h>

int test(char *URL)
{
  CURLM *handle;
  int res = CURLE_OK;
  static const char * const bl_servers[] =
     {"Microsoft-IIS/6.0", "nginx/0.8.54", NULL};
  static const char * const bl_sites[] =
     {"curl.se:443", "example.com:80", NULL};

  global_init(CURL_GLOBAL_ALL);
  handle = curl_multi_init();
  (void)URL; /* unused */

  curl_multi_setopt(handle, CURLMOPT_PIPELINING_SERVER_BL, bl_servers);
  curl_multi_setopt(handle, CURLMOPT_PIPELINING_SITE_BL, bl_sites);
  curl_multi_cleanup(handle);
  curl_global_cleanup();
  return 0;
}
