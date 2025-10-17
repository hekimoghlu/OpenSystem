/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#include "oniguruma.h"

static int
search(regex_t* reg, unsigned char* str, unsigned char* end)
{
  int r;
  unsigned char *start, *range;
  OnigRegion *region;

  region = onig_region_new();

  start = str;
  range = end;
  r = onig_search(reg, str, end, start, range, region, ONIG_OPTION_NONE);
  if (r >= 0) {
    int i;

    fprintf(stderr, "match at %d  (%s)\n", r,
            ONIGENC_NAME(onig_get_encoding(reg)));
    for (i = 0; i < region->num_regs; i++) {
      fprintf(stderr, "%d: (%d-%d)\n", i, region->beg[i], region->end[i]);
    }
  }
  else if (r == ONIG_MISMATCH) {
    fprintf(stderr, "search fail (%s)\n",
            ONIGENC_NAME(onig_get_encoding(reg)));
  }
  else { /* error */
    char s[ONIG_MAX_ERROR_MESSAGE_LEN];
    onig_error_code_to_str((UChar* )s, r);
    fprintf(stderr, "ERROR: %s\n", s);
    fprintf(stderr, "  (%s)\n", ONIGENC_NAME(onig_get_encoding(reg)));
    onig_region_free(region, 1 /* 1:free self, 0:free contents only */);
    return -1;
  }

  onig_region_free(region, 1 /* 1:free self, 0:free contents only */);
  return 0;
}

static int
exec(OnigEncoding enc, OnigOptionType options, char* apattern, char* astr)
{
  int r;
  unsigned char *end;
  regex_t* reg;
  OnigErrorInfo einfo;
  UChar* pattern = (UChar* )apattern;
  UChar* str     = (UChar* )astr;

  r = onig_new(&reg, pattern,
               pattern + onigenc_str_bytelen_null(enc, pattern),
               options, enc, ONIG_SYNTAX_DEFAULT, &einfo);
  if (r != ONIG_NORMAL) {
    char s[ONIG_MAX_ERROR_MESSAGE_LEN];
    onig_error_code_to_str((UChar* )s, r, &einfo);
    fprintf(stderr, "ERROR: %s\n", s);
    return -1;
  }

  end = str + onigenc_str_bytelen_null(enc, str);
  r = search(reg, str, end);

  onig_free(reg);
  return 0;
}



extern int main(int argc, char* argv[])
{
  OnigEncoding use_encs[1];

  use_encs[0] = ONIG_ENCODING_UTF8;
  onig_initialize(use_encs, 1);

  /* fix ignore case in look-behind
     commit: 3340ec2cc5627172665303fe248c9793354d2251 */
  exec(ONIG_ENCODING_UTF8, ONIG_OPTION_IGNORECASE,
       "\305\211a", "\312\274na"); /* \u{0149}a  \u{02bc}na */

  exec(ONIG_ENCODING_UTF8, ONIG_OPTION_NONE, "(\\2)(\\1)", "aa"); /* fail. */

  exec(ONIG_ENCODING_UTF8, ONIG_OPTION_FIND_LONGEST,
       "a*", "aa aaa aaaa aaaaa "); /* match 12-17 */

  onig_end();
  return 0;
}
