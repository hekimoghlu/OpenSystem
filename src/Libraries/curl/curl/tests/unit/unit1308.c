/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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
#define CURL_DISABLE_DEPRECATION  /* Testing the form api */
#include "curlcheck.h"

#include <curl/curl.h>

static CURLcode unit_setup(void)
{
  return CURLE_OK;
}

static void unit_stop(void)
{

}

static size_t print_httppost_callback(void *arg, const char *buf, size_t len)
{
  fwrite(buf, len, 1, stdout);
  (*(size_t *) arg) += len;
  return len;
}

UNITTEST_START
  int rc;
  struct curl_httppost *post = NULL;
  struct curl_httppost *last = NULL;
  size_t total_size = 0;
  char buffer[] = "test buffer";

  rc = curl_formadd(&post, &last, CURLFORM_COPYNAME, "name",
                    CURLFORM_COPYCONTENTS, "content", CURLFORM_END);

  fail_unless(rc == 0, "curl_formadd returned error");

  /* after the first curl_formadd when there's a single entry, both pointers
     should point to the same struct */
  fail_unless(post == last, "post and last weren't the same");

  rc = curl_formadd(&post, &last, CURLFORM_COPYNAME, "htmlcode",
                    CURLFORM_COPYCONTENTS, "<HTML></HTML>",
                    CURLFORM_CONTENTTYPE, "text/html", CURLFORM_END);

  fail_unless(rc == 0, "curl_formadd returned error");

  rc = curl_formadd(&post, &last, CURLFORM_COPYNAME, "name_for_ptrcontent",
                   CURLFORM_PTRCONTENTS, buffer, CURLFORM_END);

  fail_unless(rc == 0, "curl_formadd returned error");

  rc = curl_formget(post, &total_size, print_httppost_callback);

  fail_unless(rc == 0, "curl_formget returned error");

  fail_unless(total_size == 518, "curl_formget got wrong size back");

  curl_formfree(post);

  /* start a new formpost with a file upload and formget */
  post = last = NULL;

  rc = curl_formadd(&post, &last,
                    CURLFORM_PTRNAME, "name of file field",
                    CURLFORM_FILE, arg,
                    CURLFORM_FILENAME, "custom named file",
                    CURLFORM_END);

  fail_unless(rc == 0, "curl_formadd returned error");

  rc = curl_formget(post, &total_size, print_httppost_callback);
  fail_unless(rc == 0, "curl_formget returned error");
  fail_unless(total_size == 899, "curl_formget got wrong size back");

  curl_formfree(post);

UNITTEST_STOP
