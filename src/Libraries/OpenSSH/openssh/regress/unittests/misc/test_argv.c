/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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
/*
 * Regress test for misc argv handling functions.
 *
 * Placed in the public domain.
 */

#include "includes.h"

#include <sys/types.h>
#include <stdio.h>
#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif
#include <stdlib.h>
#include <string.h>

#include "../test_helper/test_helper.h"

#include "log.h"
#include "misc.h"

void test_argv(void);

void
test_argv(void)
{
	char **av = NULL;
	int ac = 0;

#define RESET_ARGV() \
	do { \
		argv_free(av, ac); \
		av = NULL; \
		ac = -1; \
	} while (0)

	TEST_START("empty args");
	ASSERT_INT_EQ(argv_split("", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 0);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_PTR_EQ(av[0], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("    ", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 0);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_PTR_EQ(av[0], NULL);
	RESET_ARGV();
	TEST_DONE();

	TEST_START("trivial args");
	ASSERT_INT_EQ(argv_split("leamas", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 1);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "leamas");
	ASSERT_PTR_EQ(av[1], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("smiley leamas", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 2);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "smiley");
	ASSERT_STRING_EQ(av[1], "leamas");
	ASSERT_PTR_EQ(av[2], NULL);
	RESET_ARGV();
	TEST_DONE();

	TEST_START("quoted");
	ASSERT_INT_EQ(argv_split("\"smiley\"", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 1);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "smiley");
	ASSERT_PTR_EQ(av[1], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("leamas \" smiley \"", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 2);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "leamas");
	ASSERT_STRING_EQ(av[1], " smiley ");
	ASSERT_PTR_EQ(av[2], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("\"smiley leamas\"", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 1);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "smiley leamas");
	ASSERT_PTR_EQ(av[1], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("smiley\" leamas\" liz", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 2);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "smiley leamas");
	ASSERT_STRING_EQ(av[1], "liz");
	ASSERT_PTR_EQ(av[2], NULL);
	RESET_ARGV();
	TEST_DONE();

	TEST_START("escaped");
	ASSERT_INT_EQ(argv_split("\\\"smiley\\'", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 1);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "\"smiley'");
	ASSERT_PTR_EQ(av[1], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("'\\'smiley\\\"'", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 1);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "'smiley\"");
	ASSERT_PTR_EQ(av[1], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("smiley\\'s leamas\\'", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 2);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "smiley's");
	ASSERT_STRING_EQ(av[1], "leamas'");
	ASSERT_PTR_EQ(av[2], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("leamas\\\\smiley", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 1);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "leamas\\smiley");
	ASSERT_PTR_EQ(av[1], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("leamas\\\\ \\\\smiley", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 2);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "leamas\\");
	ASSERT_STRING_EQ(av[1], "\\smiley");
	ASSERT_PTR_EQ(av[2], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("smiley\\ leamas", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 1);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "smiley leamas");
	ASSERT_PTR_EQ(av[1], NULL);
	RESET_ARGV();
	TEST_DONE();

	TEST_START("quoted escaped");
	ASSERT_INT_EQ(argv_split("'smiley\\ leamas'", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 1);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "smiley\\ leamas");
	ASSERT_PTR_EQ(av[1], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("\"smiley\\ leamas\"", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 1);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "smiley\\ leamas");
	ASSERT_PTR_EQ(av[1], NULL);
	RESET_ARGV();
	TEST_DONE();

	TEST_START("comments");
	ASSERT_INT_EQ(argv_split("# gold", &ac, &av, 0), 0);
	ASSERT_INT_EQ(ac, 2);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "#");
	ASSERT_STRING_EQ(av[1], "gold");
	ASSERT_PTR_EQ(av[2], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("# gold", &ac, &av, 1), 0);
	ASSERT_INT_EQ(ac, 0);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_PTR_EQ(av[0], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("leamas#gold", &ac, &av, 1), 0);
	ASSERT_INT_EQ(ac, 1);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "leamas#gold");
	ASSERT_PTR_EQ(av[1], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("\"leamas # gold\"", &ac, &av, 1), 0);
	ASSERT_INT_EQ(ac, 1);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "leamas # gold");
	ASSERT_PTR_EQ(av[1], NULL);
	RESET_ARGV();
	ASSERT_INT_EQ(argv_split("\"leamas\"#gold", &ac, &av, 1), 0);
	ASSERT_INT_EQ(ac, 1);
	ASSERT_PTR_NE(av, NULL);
	ASSERT_STRING_EQ(av[0], "leamas#gold");
	ASSERT_PTR_EQ(av[1], NULL);
	RESET_ARGV();
	TEST_DONE();

	/* XXX test char *argv_assemble(int argc, char **argv) */
}
