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
#include "expat_config.h"

#include <stdio.h>
#include <string.h>

#include "expat.h"
#include "internal.h"
#include "minicheck.h"
#include "common.h"

#include "basic_tests.h"
#include "ns_tests.h"
#include "misc_tests.h"
#include "alloc_tests.h"
#include "nsalloc_tests.h"
#include "acc_tests.h"

XML_Parser g_parser = NULL;

static Suite *
make_suite(void) {
  Suite *s = suite_create("basic");

  make_basic_test_case(s);
  make_namespace_test_case(s);
  make_miscellaneous_test_case(s);
  make_alloc_test_case(s);
  make_nsalloc_test_case(s);
#if XML_GE == 1
  make_accounting_test_case(s);
#endif

  return s;
}

int
main(int argc, char *argv[]) {
  int i, nf;
  int verbosity = CK_NORMAL;
  Suite *s = make_suite();
  SRunner *sr = srunner_create(s);

  for (i = 1; i < argc; ++i) {
    char *opt = argv[i];
    if (strcmp(opt, "-v") == 0 || strcmp(opt, "--verbose") == 0)
      verbosity = CK_VERBOSE;
    else if (strcmp(opt, "-q") == 0 || strcmp(opt, "--quiet") == 0)
      verbosity = CK_SILENT;
    else {
      fprintf(stderr, "runtests: unknown option '%s'\n", opt);
      return 2;
    }
  }
  if (verbosity != CK_SILENT)
    printf("Expat version: %" XML_FMT_STR "\n", XML_ExpatVersion());

  for (g_chunkSize = 0; g_chunkSize <= 5; g_chunkSize++) {
    for (int enabled = 0; enabled <= 1; ++enabled) {
      char context[100];
      g_reparseDeferralEnabledDefault = enabled;
      snprintf(context, sizeof(context), "chunksize=%d deferral=%d",
               g_chunkSize, enabled);
      context[sizeof(context) - 1] = '\0';
      srunner_run_all(sr, context, verbosity);
    }
  }
  srunner_summarize(sr, verbosity);
  nf = srunner_ntests_failed(sr);
  srunner_free(sr);

  return (nf == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

