/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 2, 2023.
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

#include "bufref.h"

static struct bufref bufref;

static int freecount = 0;

static void test_free(void *p)
{
  fail_unless(p, "pointer to free may not be NULL");
  freecount++;
  free(p);
}

static CURLcode unit_setup(void)
{
  Curl_bufref_init(&bufref);
  return CURLE_OK;
}

static void unit_stop(void)
{
}

UNITTEST_START
{
  char *buffer = NULL;
  CURLcode result = CURLE_OK;

  /**
   * testing Curl_bufref_init.
   * @assumptions:
   * 1: data size will be 0
   * 2: reference will be NULL
   * 3: destructor will be NULL
   */

  fail_unless(!bufref.ptr, "Initial reference must be NULL");
  fail_unless(!bufref.len, "Initial length must be NULL");
  fail_unless(!bufref.dtor, "Destructor must be NULL");

  /**
   * testing Curl_bufref_set
   */

  buffer = malloc(13);
  abort_unless(buffer, "Out of memory");
  Curl_bufref_set(&bufref, buffer, 13, test_free);

  fail_unless((char *) bufref.ptr == buffer, "Referenced data badly set");
  fail_unless(bufref.len == 13, "Data size badly set");
  fail_unless(bufref.dtor == test_free, "Destructor badly set");

  /**
   * testing Curl_bufref_ptr
   */

  fail_unless((char *) Curl_bufref_ptr(&bufref) == buffer,
              "Wrong pointer value returned");

  /**
   * testing Curl_bufref_len
   */

  fail_unless(Curl_bufref_len(&bufref) == 13, "Wrong data size returned");

  /**
   * testing Curl_bufref_memdup
   */

  result = Curl_bufref_memdup(&bufref, "1661", 3);
  abort_unless(result == CURLE_OK, curl_easy_strerror(result));
  fail_unless(freecount == 1, "Destructor not called");
  fail_unless((char *) bufref.ptr != buffer, "Returned pointer not set");
  buffer = (char *) Curl_bufref_ptr(&bufref);
  fail_unless(buffer, "Allocated pointer is NULL");
  fail_unless(bufref.len == 3, "Wrong data size stored");
  fail_unless(!buffer[3], "Duplicated data should have been truncated");
  fail_unless(!strcmp(buffer, "166"), "Bad duplicated data");

  /**
   * testing Curl_bufref_free
   */

  Curl_bufref_free(&bufref);
  fail_unless(freecount == 1, "Wrong destructor called");
  fail_unless(!bufref.ptr, "Initial reference must be NULL");
  fail_unless(!bufref.len, "Initial length must be NULL");
  fail_unless(!bufref.dtor, "Destructor must be NULL");
}
UNITTEST_STOP
