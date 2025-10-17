/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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

#include <darwintest.h>
#include <stdlib.h>
#include <os/stdlib.h>

T_DECL(os_simple_hash, "sanity check of os_simple_hash",
		T_META_ALL_VALID_ARCHS(true))
{
	const char * string =
			"We made the buttons on the screen look so good you'll want to lick them.";
	uint64_t hashval = os_simple_hash_string(string);
	T_EXPECT_NE(hashval, 0ULL, "usually should get a non-0 hash value");

	char buf[1024];
	arc4random_buf(buf, sizeof(buf));
	hashval = os_simple_hash(buf, sizeof(buf));
	T_EXPECT_NE(hashval, 0ULL, "usually should get a non-0 hash value");
}

T_DECL(os_simple_hash_seeds, "os_simple_hash different seeds give different hashes",
		T_META_ALL_VALID_ARCHS(true))
{
	const char * string =
			"We made the buttons on the screen look so good you'll want to lick them.";

	uint64_t hashval0 = os_simple_hash_string_with_seed(string, 0x0);
	T_EXPECT_NE(hashval0, 0ULL, "usually should get a non-0 hash value");
	uint64_t hashval1 = os_simple_hash_string_with_seed(string, 0x1);
	T_EXPECT_NE(hashval1, 0ULL, "usually should get a non-0 hash value");
	uint64_t hashvalF = os_simple_hash_string_with_seed(string, 0xF);
	T_EXPECT_NE(hashvalF, 0ULL, "usually should get a non-0 hash value");
	uint64_t hashvalFoo = os_simple_hash_string_with_seed(string, 0xF0000000);
	T_EXPECT_NE(hashvalFoo, 0ULL, "usually should get a non-0 hash value");

	T_EXPECT_NE(hashval0, hashval1, NULL);
	T_EXPECT_NE(hashval0, hashvalF, NULL);
	T_EXPECT_NE(hashval0, hashvalFoo, NULL);
	T_EXPECT_NE(hashval1, hashvalF, NULL);
	T_EXPECT_NE(hashval1, hashvalFoo, NULL);
	T_EXPECT_NE(hashvalF, hashvalFoo, NULL);
}
