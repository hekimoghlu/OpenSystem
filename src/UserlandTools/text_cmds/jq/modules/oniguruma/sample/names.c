/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 27, 2022.
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
#include <stdio.h>
#include <string.h>
#include "oniguruma.h"

static int
name_callback(const UChar* name, const UChar* name_end,
              int ngroup_num, int* group_nums,
              regex_t* reg, void* arg)
{
  int i, gn, ref;
  char* s;
  OnigRegion *region = (OnigRegion* )arg;

  for (i = 0; i < ngroup_num; i++) {
    gn = group_nums[i];
    ref = onig_name_to_backref_number(reg, name, name_end, region);
    s = (ref == gn ? "*" : "");
    fprintf(stderr, "%s (%d): ", name, gn);
    fprintf(stderr, "(%d-%d) %s\n", region->beg[gn], region->end[gn], s);
  }
  return 0;  /* 0: continue */
}

extern int main(int argc, char* argv[])
{
  int r;
  unsigned char *start, *range, *end;
  regex_t* reg;
  OnigErrorInfo einfo;
  OnigRegion *region;
  OnigEncoding use_encs[1];

  static UChar* pattern = (UChar* )"(?<foo>a*)(?<bar>b*)(?<foo>c*)";
  static UChar* str = (UChar* )"aaabbbbcc";

  use_encs[0] = ONIG_ENCODING_ASCII;
  onig_initialize(use_encs, sizeof(use_encs)/sizeof(use_encs[0]));

  r = onig_new(&reg, pattern, pattern + strlen((char* )pattern),
        ONIG_OPTION_DEFAULT, ONIG_ENCODING_ASCII, ONIG_SYNTAX_DEFAULT, &einfo);
  if (r != ONIG_NORMAL) {
    char s[ONIG_MAX_ERROR_MESSAGE_LEN];
    onig_error_code_to_str((UChar* )s, r, &einfo);
    fprintf(stderr, "ERROR: %s\n", s);
    return -1;
  }

  fprintf(stderr, "number of names: %d\n", onig_number_of_names(reg));

  region = onig_region_new();

  end   = str + strlen((char* )str);
  start = str;
  range = end;
  r = onig_search(reg, str, end, start, range, region, ONIG_OPTION_NONE);
  if (r >= 0) {
    fprintf(stderr, "match at %d\n\n", r);
    r = onig_foreach_name(reg, name_callback, (void* )region);
  }
  else if (r == ONIG_MISMATCH) {
    fprintf(stderr, "search fail\n");
  }
  else { /* error */
    char s[ONIG_MAX_ERROR_MESSAGE_LEN];
    onig_error_code_to_str((UChar* )s, r);
    onig_region_free(region, 1 /* 1:free self, 0:free contents only */);
    onig_free(reg);
    onig_end();
    return -1;
  }

  onig_region_free(region, 1 /* 1:free self, 0:free contents only */);
  onig_free(reg);
  onig_end();
  return 0;
}
