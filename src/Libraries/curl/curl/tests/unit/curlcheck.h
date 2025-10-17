/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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

/* The fail macros mark the current test step as failed, and continue */
#define fail_if(expr, msg)                                       \
  do {                                                           \
    if(expr) {                                                   \
      fprintf(stderr, "%s:%d FAILED Assertion '%s' met: %s\n",   \
              __FILE__, __LINE__, #expr, msg);                   \
      unitfail++;                                                \
    }                                                            \
  } while(0)

#define fail_unless(expr, msg)                             \
  do {                                                     \
    if(!(expr)) {                                          \
      fprintf(stderr, "%s:%d Assertion '%s' FAILED: %s\n", \
              __FILE__, __LINE__, #expr, msg);             \
      unitfail++;                                          \
    }                                                      \
  } while(0)

#define verify_memory(dynamic, check, len)                              \
  do {                                                                  \
    if(dynamic && memcmp(dynamic, check, len)) {                        \
      fprintf(stderr, "%s:%d Memory buffer FAILED match size %d. "      \
              "'%s' is not\n", __FILE__, __LINE__, len,                 \
              hexdump((const unsigned char *)check, len));              \
      fprintf(stderr, "%s:%d the same as '%s'\n", __FILE__, __LINE__,   \
              hexdump((const unsigned char *)dynamic, len));            \
      unitfail++;                                                       \
    }                                                                   \
  } while(0)

/* fail() is for when the test case figured out by itself that a check
   proved a failure */
#define fail(msg) do {                                                 \
    fprintf(stderr, "%s:%d test FAILED: '%s'\n",                       \
            __FILE__, __LINE__, msg);                                  \
    unitfail++;                                                        \
  } while(0)


/* The abort macros mark the current test step as failed, and exit the test */
#define abort_if(expr, msg)                                     \
  do {                                                          \
    if(expr) {                                                  \
      fprintf(stderr, "%s:%d ABORT assertion '%s' met: %s\n",   \
              __FILE__, __LINE__, #expr, msg);                  \
      unitfail++;                                               \
      goto unit_test_abort;                                     \
    }                                                           \
  } while(0)

#define abort_unless(expr, msg)                                         \
  do {                                                                  \
    if(!(expr)) {                                                       \
      fprintf(stderr, "%s:%d ABORT assertion '%s' failed: %s\n",        \
              __FILE__, __LINE__, #expr, msg);                          \
      unitfail++;                                                       \
      goto unit_test_abort;                                             \
    }                                                                   \
  } while(0)

#define abort_test(msg)                                       \
  do {                                                        \
    fprintf(stderr, "%s:%d test ABORTED: '%s'\n",             \
            __FILE__, __LINE__, msg);                         \
    unitfail++;                                               \
    goto unit_test_abort;                                     \
  } while(0)


#define UNITTEST_START                          \
  int test(char *arg)                           \
  {                                             \
    (void)arg;                                  \
    if(unit_setup()) {                          \
      fail("unit_setup() FAILURE");             \
    }                                           \
    else {

#define UNITTEST_STOP                           \
    goto unit_test_abort; /* avoid warning */   \
unit_test_abort:                                \
    unit_stop();                                \
  }                                             \
  return unitfail;                              \
  }
