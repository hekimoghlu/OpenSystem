/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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
#include "onigposix.h"

typedef unsigned char  UChar;

static int x(regex_t* reg, unsigned char* pattern, unsigned char* str)
{
  int r, i;
  char buf[200];
  regmatch_t pmatch[20];

  r = regexec(reg, (char* )str, reg->re_nsub + 1, pmatch, 0);
  if (r != 0 && r != REG_NOMATCH) {
    regerror(r, reg, buf, sizeof(buf));
    fprintf(stderr, "ERROR: %s\n", buf);
    regfree(reg);
    return -1;
  }

  if (r == REG_NOMATCH) {
    fprintf(stderr, "FAIL: /%s/ '%s'\n", pattern, str);
  }
  else {
    fprintf(stderr, "OK: /%s/ '%s'\n", pattern, str);
    for (i = 0; i <= (int )reg->re_nsub; i++) {
      fprintf(stderr, "%d: %d-%d\n", i, pmatch[i].rm_so, pmatch[i].rm_eo);
    }
  }
  regfree(reg);
  return 0;
}

extern int main(int argc, char* argv[])
{
  int r;
  char buf[200];
  regex_t reg;
  UChar* pattern;

  reg_set_encoding(REG_POSIX_ENCODING_ASCII);

  /* default syntax (ONIG_SYNTAX_ONIGURUMA) */
  pattern = (UChar* )"^a+b{2,7}[c-f]?$|uuu";
  r = regcomp(&reg, (char* )pattern, REG_EXTENDED);
  if (r) {
    regerror(r, &reg, buf, sizeof(buf));
    fprintf(stderr, "ERROR: %s\n", buf);
    regfree(&reg);
    onig_end();
    return -1;
  }
  x(&reg, pattern, (UChar* )"aaabbbbd");

  /* POSIX Basic RE (REG_EXTENDED is not specified.) */
  pattern = (UChar* )"^a+b{2,7}[c-f]?|uuu";
  r = regcomp(&reg, (char* )pattern, 0);
  if (r) {
    regerror(r, &reg, buf, sizeof(buf));
    fprintf(stderr, "ERROR: %s\n", buf);
    regfree(&reg);
    onig_end();
    return -1;
  }
  x(&reg, pattern, (UChar* )"a+b{2,7}d?|uuu");

  /* POSIX Basic RE (REG_EXTENDED is not specified.) */
  pattern = (UChar* )"^a*b\\{2,7\\}\\([c-f]\\)$";
  r = regcomp(&reg, (char* )pattern, 0);
  if (r) {
    regerror(r, &reg, buf, sizeof(buf));
    fprintf(stderr, "ERROR: %s\n", buf);
    regfree(&reg);
    onig_end();
    return -1;
  }
  x(&reg, pattern, (UChar* )"aaaabbbbbbd");

  /* POSIX Extended RE */
  onig_set_default_syntax(ONIG_SYNTAX_POSIX_EXTENDED);
  pattern = (UChar* )"^a+b{2,7}[c-f]?)$|uuu";
  r = regcomp(&reg, (char* )pattern, REG_EXTENDED);
  if (r) {
    regerror(r, &reg, buf, sizeof(buf));
    fprintf(stderr, "ERROR: %s\n", buf);
    regfree(&reg);
    onig_end();
    return -1;
  }
  x(&reg, pattern, (UChar* )"aaabbbbd)");

  pattern = (UChar* )"^b.";
  r = regcomp(&reg, (char* )pattern, REG_EXTENDED | REG_NEWLINE);
  if (r) {
    regerror(r, &reg, buf, sizeof(buf));
    fprintf(stderr, "ERROR: %s\n", buf);
    regfree(&reg);
    onig_end();
    return -1;
  }
  x(&reg, pattern, (UChar* )"a\nb\n");

  onig_end();
  return 0;
}
