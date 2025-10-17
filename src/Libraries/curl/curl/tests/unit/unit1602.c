/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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

#define ENABLE_CURLX_PRINTF
#include "curlx.h"

#include "hash.h"

#include "memdebug.h" /* LAST include file */

static struct Curl_hash hash_static;

static void mydtor(void *p)
{
  int *ptr = (int *)p;
  free(ptr);
}

static CURLcode unit_setup(void)
{
  Curl_hash_init(&hash_static, 7, Curl_hash_str,
                 Curl_str_key_compare, mydtor);
  return CURLE_OK;
}

static void unit_stop(void)
{
  Curl_hash_destroy(&hash_static);
}

UNITTEST_START
  int *value;
  int *value2;
  int *nodep;
  size_t klen = sizeof(int);

  int key = 20;
  int key2 = 25;


  value = malloc(sizeof(int));
  abort_unless(value != NULL, "Out of memory");
  *value = 199;
  nodep = Curl_hash_add(&hash_static, &key, klen, value);
  if(!nodep)
    free(value);
  abort_unless(nodep, "insertion into hash failed");
  Curl_hash_clean(&hash_static);

  /* Attempt to add another key/value pair */
  value2 = malloc(sizeof(int));
  abort_unless(value2 != NULL, "Out of memory");
  *value2 = 204;
  nodep = Curl_hash_add(&hash_static, &key2, klen, value2);
  if(!nodep)
    free(value2);
  abort_unless(nodep, "insertion into hash failed");

UNITTEST_STOP
