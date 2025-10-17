/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
#include "curl_setup.h"
#include "strcase.h"
#include "easyoptions.h"

#ifndef CURL_DISABLE_GETOPTIONS

/* Lookups easy options at runtime */
static struct curl_easyoption *lookup(const char *name, CURLoption id)
{
  DEBUGASSERT(name || id);
  DEBUGASSERT(!Curl_easyopts_check());
  if(name || id) {
    struct curl_easyoption *o = &Curl_easyopts[0];
    do {
      if(name) {
        if(strcasecompare(o->name, name))
          return o;
      }
      else {
        if((o->id == id) && !(o->flags & CURLOT_FLAG_ALIAS))
          /* don't match alias options */
          return o;
      }
      o++;
    } while(o->name);
  }
  return NULL;
}

const struct curl_easyoption *curl_easy_option_by_name(const char *name)
{
  /* when name is used, the id argument is ignored */
  return lookup(name, CURLOPT_LASTENTRY);
}

const struct curl_easyoption *curl_easy_option_by_id(CURLoption id)
{
  return lookup(NULL, id);
}

/* Iterates over available options */
const struct curl_easyoption *
curl_easy_option_next(const struct curl_easyoption *prev)
{
  if(prev && prev->name) {
    prev++;
    if(prev->name)
      return prev;
  }
  else if(!prev)
    return &Curl_easyopts[0];
  return NULL;
}

#else
const struct curl_easyoption *curl_easy_option_by_name(const char *name)
{
  (void)name;
  return NULL;
}

const struct curl_easyoption *curl_easy_option_by_id (CURLoption id)
{
  (void)id;
  return NULL;
}

const struct curl_easyoption *
curl_easy_option_next(const struct curl_easyoption *prev)
{
  (void)prev;
  return NULL;
}
#endif
