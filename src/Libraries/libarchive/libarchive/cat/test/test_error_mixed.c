/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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

DEFINE_TEST(test_error_mixed)
{
	const char *reffile1 = "test_expand.plain";
	const char *reffile2 = "test_expand.error";
	const char *reffile3 = "test_expand.Z";

	assertFileNotExists(reffile2);
	extract_reference_file(reffile1);
	extract_reference_file(reffile3);
	assert(0 != systemf("%s %s %s %s >test.out 2>test.err",
	    testprog, reffile1, reffile2, reffile3));

	assertTextFileContents(
	    "contents of test_expand.plain.\n"
	    "contents of test_expand.Z.\n", "test.out");
	assertNonEmptyFile("test.err");
}
