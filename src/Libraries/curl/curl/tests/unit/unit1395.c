/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 9, 2025.
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

/* copied from urlapi.c */
extern int dedotdotify(const char *input, size_t clen, char **out);

#include "memdebug.h"

static CURLcode unit_setup(void)
{
  return CURLE_OK;
}

static void unit_stop(void)
{

}

struct dotdot {
  const char *input;
  const char *output;
};

UNITTEST_START

  unsigned int i;
  int fails = 0;
  const struct dotdot pairs[] = {
    { "/a/b/c/./../../g", "/a/g" },
    { "mid/content=5/../6", "mid/6" },
    { "/hello/../moo", "/moo" },
    { "/1/../1", "/1" },
    { "/1/./1", "/1/1" },
    { "/1/..", "/" },
    { "/1/.", "/1/" },
    { "/1/./..", "/" },
    { "/1/./../2", "/2" },
    { "/hello/1/./../2", "/hello/2" },
    { "test/this", NULL },
    { "test/this/../now", "test/now" },
    { "/1../moo../foo", "/1../moo../foo"},
    { "/../../moo", "/moo"},
    { "/../../moo?", "/moo?"},
    { "/123?", NULL},
    { "/../moo/..?", "/" },
    { "/", NULL },
    { "", NULL },
    { "/.../", "/.../" },
    { "./moo", "moo" },
    { "../moo", "moo" },
    { "/.", "/" },
    { "/..", "/" },
    { "/moo/..", "/" },
    { "/..", "/" },
    { "/.", "/" },
  };

  for(i = 0; i < sizeof(pairs)/sizeof(pairs[0]); i++) {
    char *out;
    int err = dedotdotify(pairs[i].input, strlen(pairs[i].input), &out);
    abort_unless(err == 0, "returned error");
    abort_if(err && out, "returned error with output");

    if(out && pairs[i].output && strcmp(out, pairs[i].output)) {
      fprintf(stderr, "Test %u: '%s' gave '%s' instead of '%s'\n",
              i, pairs[i].input, out, pairs[i].output);
      fail("Test case output mismatched");
      fails++;
    }
    else if((!out && pairs[i].output) ||
            (out && !pairs[i].output)) {
      fprintf(stderr, "Test %u: '%s' gave '%s' instead of '%s'\n",
              i, pairs[i].input, out ? out : "(null)",
              pairs[i].output ? pairs[i].output : "(null)");
      fail("Test case output mismatched");
      fails++;
    }
    else
      fprintf(stderr, "Test %u: OK\n", i);
    free(out);
  }

  fail_if(fails, "output mismatched");

UNITTEST_STOP
