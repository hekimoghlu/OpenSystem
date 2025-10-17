/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
 * Outputs all protocols and features supported
 * </DESC>
 */
#include <stdio.h>
#include <curl/curl.h>

#if !CURL_AT_LEAST_VERSION(7,87,0)
#error "too old libcurl"
#endif

int main(void)
{
  curl_version_info_data *ver;
  const char *const *ptr;

  curl_global_init(CURL_GLOBAL_ALL);

  ver = curl_version_info(CURLVERSION_NOW);
  printf("Protocols:\n");
  for(ptr = ver->protocols; *ptr; ++ptr)
    printf("  %s\n", *ptr);
  printf("Features:\n");
  for(ptr = ver->feature_names; *ptr; ++ptr)
    printf("  %s\n", *ptr);

  curl_global_cleanup();
  return 0;
}
