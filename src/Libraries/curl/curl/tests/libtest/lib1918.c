/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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

int test(char *URL)
{
  const struct curl_easyoption *o;
  int error = 0;
  (void)URL;

  curl_global_init(CURL_GLOBAL_ALL);

  for(o = curl_easy_option_next(NULL);
      o;
      o = curl_easy_option_next(o)) {
    const struct curl_easyoption *ename =
      curl_easy_option_by_name(o->name);
    const struct curl_easyoption *eid =
      curl_easy_option_by_id(o->id);

    if(ename->id != o->id) {
      printf("name lookup id %d doesn't match %d\n",
             ename->id, o->id);
    }
    else if(eid->id != o->id) {
      printf("ID lookup %d doesn't match %d\n",
             ename->id, o->id);
    }
  }
  curl_global_cleanup();
  return error;
}
