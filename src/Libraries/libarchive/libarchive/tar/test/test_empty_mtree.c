/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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

/*
 * Regression test:  We used to get a bogus error message when we
 * asked tar to copy entries out of an empty archive.  See
 * Issue 51 on libarchive.googlecode.com for details.
 */
DEFINE_TEST(test_empty_mtree)
{
	int r;

	assertMakeFile("test1.mtree", 0777, "#mtree\n");

	r = systemf("%s cf test1.tar @test1.mtree >test1.out 2>test1.err",
	    testprog);
	failure("Error invoking %s cf", testprog);
	assertEqualInt(r, 0);
	assertEmptyFile("test1.out");
	assertEmptyFile("test1.err");
}
