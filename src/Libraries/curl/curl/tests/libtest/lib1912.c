/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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

#define print_err(name, exp) \
  fprintf(stderr, "Type mismatch for CURLOPT_%s (expected %s)\n", name, exp);

int test(char *URL)
{
/* Only test if GCC typechecking is available */
  int error = 0;
#ifdef CURLINC_TYPECHECK_GCC_H
  const struct curl_easyoption *o;
  for(o = curl_easy_option_next(NULL);
      o;
      o = curl_easy_option_next(o)) {
    CURL_IGNORE_DEPRECATION(
      /* Test for mismatch OR missing typecheck macros */
      if(curlcheck_long_option(o->id) !=
          (o->type == CURLOT_LONG || o->type == CURLOT_VALUES)) {
        print_err(o->name, "CURLOT_LONG or CURLOT_VALUES");
        error++;
      }
      if(curlcheck_off_t_option(o->id) != (o->type == CURLOT_OFF_T)) {
        print_err(o->name, "CURLOT_OFF_T");
        error++;
      }
      if(curlcheck_string_option(o->id) != (o->type == CURLOT_STRING)) {
        print_err(o->name, "CURLOT_STRING");
        error++;
      }
      if(curlcheck_slist_option(o->id) != (o->type == CURLOT_SLIST)) {
        print_err(o->name, "CURLOT_SLIST");
        error++;
      }
      if(curlcheck_cb_data_option(o->id) != (o->type == CURLOT_CBPTR)) {
        print_err(o->name, "CURLOT_CBPTR");
        error++;
      }
      /* From here: only test that the type matches if macro is known */
      if(curlcheck_write_cb_option(o->id) && (o->type != CURLOT_FUNCTION)) {
        print_err(o->name, "CURLOT_FUNCTION");
        error++;
      }
      if(curlcheck_conv_cb_option(o->id) && (o->type != CURLOT_FUNCTION)) {
        print_err(o->name, "CURLOT_FUNCTION");
        error++;
      }
      if(curlcheck_postfields_option(o->id) && (o->type != CURLOT_OBJECT)) {
        print_err(o->name, "CURLOT_OBJECT");
        error++;
      }
      /* Todo: no gcc typecheck for CURLOPTTYPE_BLOB types? */
    )
  }
#endif
  (void)URL;
  return error;
}
