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

#include "urldata.h"
#include "dynbuf.h"
#include "dynhds.h"
#include "curl_trc.h"

static CURLcode unit_setup(void)
{
  return CURLE_OK;
}

static void unit_stop(void)
{
}


UNITTEST_START

  struct dynhds hds;
  struct dynbuf dbuf;
  CURLcode result;
  size_t i;

  /* add 1 more header than allowed */
  Curl_dynhds_init(&hds, 2, 128);
  fail_if(Curl_dynhds_count(&hds), "should be empty");
  fail_if(Curl_dynhds_add(&hds, "test1", 5, "123", 3), "add failed");
  fail_if(Curl_dynhds_add(&hds, "test2", 5, "456", 3), "add failed");
  /* remove and add without exceeding limits */
  for(i = 0; i < 100; ++i) {
    if(Curl_dynhds_remove(&hds, "test2", 5) != 1) {
      fail_if(TRUE, "should");
      break;
    }
    if(Curl_dynhds_add(&hds, "test2", 5, "456", 3)) {
      fail_if(TRUE, "add failed");
      break;
    }
  }
  fail_unless(Curl_dynhds_count(&hds) == 2, "should hold 2");
  /* set, replacing previous entry without exceeding limits */
  for(i = 0; i < 100; ++i) {
    if(Curl_dynhds_set(&hds, "test2", 5, "456", 3)) {
      fail_if(TRUE, "add failed");
      break;
    }
  }
  fail_unless(Curl_dynhds_count(&hds) == 2, "should hold 2");
  /* exceed limit on # of entries */
  result = Curl_dynhds_add(&hds, "test3", 5, "789", 3);
  fail_unless(result, "add should have failed");

  fail_unless(Curl_dynhds_count_name(&hds, "test", 4) == 0, "false positive");
  fail_unless(Curl_dynhds_count_name(&hds, "test1", 4) == 0, "false positive");
  fail_if(Curl_dynhds_get(&hds, "test1", 4), "false positive");
  fail_unless(Curl_dynhds_get(&hds, "test1", 5), "false negative");
  fail_unless(Curl_dynhds_count_name(&hds, "test1", 5) == 1, "should");
  fail_unless(Curl_dynhds_ccount_name(&hds, "test2") == 1, "should");
  fail_unless(Curl_dynhds_cget(&hds, "test2"), "should");
  fail_unless(Curl_dynhds_ccount_name(&hds, "TEST2") == 1, "should");
  fail_unless(Curl_dynhds_ccontains(&hds, "TesT2"), "should");
  fail_unless(Curl_dynhds_contains(&hds, "TeSt2", 5), "should");
  Curl_dynhds_free(&hds);

  /* add header exceeding max overall length */
  Curl_dynhds_init(&hds, 128, 10);
  fail_if(Curl_dynhds_add(&hds, "test1", 5, "123", 3), "add failed");
  fail_unless(Curl_dynhds_add(&hds, "test2", 5, "456", 3), "should fail");
  fail_if(Curl_dynhds_add(&hds, "t", 1, "1", 1), "add failed");
  Curl_dynhds_reset(&hds);
  Curl_dynhds_free(&hds);

  Curl_dynhds_init(&hds, 128, 4*1024);
  fail_if(Curl_dynhds_add(&hds, "test1", 5, "123", 3), "add failed");
  fail_if(Curl_dynhds_add(&hds, "test1", 5, "123", 3), "add failed");
  fail_if(Curl_dynhds_cadd(&hds, "blablabla", "thingies"), "add failed");
  fail_if(Curl_dynhds_h1_cadd_line(&hds, "blablabla: thingies"), "add failed");
  fail_unless(Curl_dynhds_ccount_name(&hds, "blablabla") == 2, "should");
  fail_unless(Curl_dynhds_cremove(&hds, "blablabla") == 2, "should");
  fail_if(Curl_dynhds_ccontains(&hds, "blablabla"), "should not");

  result = Curl_dynhds_h1_cadd_line(&hds, "blablabla thingies");
  fail_unless(result, "add should have failed");
  if(!result) {
    fail_unless(Curl_dynhds_ccount_name(&hds, "bLABlaBlA") == 0, "should");
    fail_if(Curl_dynhds_cadd(&hds, "Bla-Bla", "thingies"), "add failed");

    Curl_dyn_init(&dbuf, 32*1024);
    fail_if(Curl_dynhds_h1_dprint(&hds, &dbuf), "h1 print failed");
    if(Curl_dyn_ptr(&dbuf)) {
      fail_if(strcmp(Curl_dyn_ptr(&dbuf),
                     "test1: 123\r\ntest1: 123\r\nBla-Bla: thingies\r\n"),
                     "h1 format differs");
    }
    Curl_dyn_free(&dbuf);
  }

  Curl_dynhds_free(&hds);
  Curl_dynhds_init(&hds, 128, 4*1024);
  /* continuation without previous header fails */
  result = Curl_dynhds_h1_cadd_line(&hds, " indented value");
  fail_unless(result, "add should have failed");

  /* continuation with previous header must succeed */
  fail_if(Curl_dynhds_h1_cadd_line(&hds, "ti1: val1"), "add");
  fail_if(Curl_dynhds_h1_cadd_line(&hds, " val2"), "add indent");
  fail_if(Curl_dynhds_h1_cadd_line(&hds, "ti2: val1"), "add");
  fail_if(Curl_dynhds_h1_cadd_line(&hds, "\tval2"), "add indent");
  fail_if(Curl_dynhds_h1_cadd_line(&hds, "ti3: val1"), "add");
  fail_if(Curl_dynhds_h1_cadd_line(&hds, "     val2"), "add indent");

  Curl_dyn_init(&dbuf, 32*1024);
  fail_if(Curl_dynhds_h1_dprint(&hds, &dbuf), "h1 print failed");
  if(Curl_dyn_ptr(&dbuf)) {
    fprintf(stderr, "indent concat: %s\n", Curl_dyn_ptr(&dbuf));
    fail_if(strcmp(Curl_dyn_ptr(&dbuf),
                   "ti1: val1 val2\r\nti2: val1 val2\r\nti3: val1 val2\r\n"),
                   "wrong format");
  }
  Curl_dyn_free(&dbuf);

  Curl_dynhds_free(&hds);

UNITTEST_STOP

