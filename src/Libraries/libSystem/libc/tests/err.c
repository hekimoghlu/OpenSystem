/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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

#include <Block.h>
#include <darwintest.h>
#include <err.h>

#define ITERATIONS 100

T_DECL(err_multiple_exit_b, "Repeated set exit blocks doesn't leak copied blocks")
{
	int __block num = 0;
	for (int i = 0; i < ITERATIONS; ++i) {
		err_set_exit_b(^(int j) { num += j; });
	}
	err_set_exit_b(NULL);
	// Dummy expect is necessary to run leaks on this test.
	T_EXPECT_NULL(NULL, "DUMMY EXPECT");
}

T_DECL(err_multiple_exit, "Setting exit w/o block after setting exit with block doesn't leak copied block")
{
	int __block num = 0;
	err_set_exit_b(^(int j) { num += j; });
	err_set_exit(NULL);
	// Dummy expect is necessary to run leaks on this test.
	T_EXPECT_NULL(NULL, "DUMMY EXPECT");
}
