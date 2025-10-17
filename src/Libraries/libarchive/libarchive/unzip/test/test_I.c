/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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

#ifdef HAVE_LOCALE_H
#include <locale.h>
#endif

/* Test I arg - file name encoding */
DEFINE_TEST(test_I)
{
	const char *reffile = "test_I.zip";
	int r;

#if HAVE_SETLOCALE
	if (NULL == setlocale(LC_ALL, "en_US.UTF-8")) {
		skipping("en_US.UTF-8 locale not available on this system.");
		return;
	}
#else
	skipping("setlocale() not available on this system.");
#endif

	extract_reference_file(reffile);
	r = systemf("%s -I UTF-8 %s >test.out 2>test.err", testprog, reffile);
	assertEqualInt(0, r);
	assertNonEmptyFile("test.out");
	assertEmptyFile("test.err");

	assertTextFileContents("Hello, World!\n", "Î“ÎµÎ¹Î¬ ÏƒÎ¿Ï… ÎšÏŒÏƒÎ¼Îµ.txt");
}
