/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 14, 2025.
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
#include <wordexp.h>
#include <stdbool.h>
#include <TargetConditionals.h>

#include <darwintest.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

#if !TARGET_OS_IPHONE
static void check(const char *words, int expected_res, const char *expected_word,
				  const char *label)
{
	wordexp_t we;
	int res = wordexp(words, &we, WRDE_NOCMD | WRDE_SHOWERR);
	T_EXPECT_EQ(res, expected_res, "%s: error code | input: %s |", label, words);
	if (res == 0 && expected_res == 0) {
		T_EXPECT_EQ(we.we_wordc, (size_t)1,
					"%s: we_wordc | input: %s |", label, words);
		if (we.we_wordc == 1) {
			T_EXPECT_EQ_STR(we.we_wordv[0], expected_word,
							"%s: we_wordv[0] | input: %s |", label, words);
		}
	}
	if (res == 0) {
		wordfree(&we);
	}
}
#endif

T_DECL(wordexp_backtick,
	   "Check that wordexp blocks backtick only when it should")
{
#if TARGET_OS_IPHONE
	T_SKIP("wordexp doesn't exist on this OS");
#else /* !TARGET_OS_IPHONE */
	check("`",               WRDE_CMDSUB, NULL,     "unquoted backtick");
	check("\\`",             0,           "`",      "unquoted escaped backtick");
	check("\\\\`",           WRDE_CMDSUB, NULL,     "unquoted escaped backslash + backtick");
	check("'`'",             0,           "`",      "single quoted backtick");
	check("\"`\"",           WRDE_CMDSUB, NULL,     "double quoted backtick");
	check("\"\\`\"",         0,           "`",      "double quoted escaped backtick");
	check("'`",              WRDE_SYNTAX, NULL,     "single quoted backtick w/o ending quote");

	check("\\`$(oh no) # `", WRDE_CMDSUB, NULL,     "previously exploitable");

	check("$(foo)",          WRDE_CMDSUB, NULL,     "unquoted $()");
	check("\\$\\(foo\\)",    0,           "$(foo)", "unquoted escaped $()");
	check("'$(foo)'",        0,           "$(foo)", "single quoted $()");
	check("\"$(foo)\"",      WRDE_CMDSUB, NULL,     "double quoted $()");
	check("\"\\$(foo)\"",    0,           "$(foo)", "double quoted escaped $()");

#endif /* !TARGET_OS_IPHONE */
}

