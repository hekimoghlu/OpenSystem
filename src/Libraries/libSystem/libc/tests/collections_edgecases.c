/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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
#include <stdlib.h>
#include <os/collections.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <darwintest.h>

T_DECL(map_edge_64, "Make sure 64 bit map edge cases work",
		T_META("owner", "Core Darwin Daemons & Tools"))
{
	os_map_64_t edge_64_map;
	__block bool got_cafebabe = false;
	__block bool got_deadbeaf = false;
	__block bool got_beebait = false;
	__block bool got_badfood = false;
	uint64_t value;

	T_LOG("Start");

	// *** BASIC 64 bit key testing ***

	os_map_init(&edge_64_map, NULL);

	T_ASSERT_EQ(os_map_count(&edge_64_map), 0, "Expect map to be empty");

	os_map_insert(&edge_64_map, 0x0, (void *)0xCAFEBABE);
	os_map_insert(&edge_64_map, 0x1, (void *)0xDEADBEEF);
	os_map_insert(&edge_64_map, 0x2, (void *)0xBEEB8);
	os_map_insert(&edge_64_map, 0x3, (void *)0xBADF00D);

	T_ASSERT_EQ(os_map_count(&edge_64_map), 4,
		    "Expect map to have 4 entries");

	os_map_foreach(&edge_64_map, ^bool (uint64_t key, void *value){
		T_LOG("Foreach called for 0x%llx, 0x%llx",
		      (unsigned long long)key, (unsigned long long)value);
		if (key == 0x0) {
			T_ASSERT_EQ(value, (void *)0xCAFEBABE,
				    "Callback expect 0xCAFEBABE");
			got_cafebabe = true;
		} else if (key == 0x1) {
			T_ASSERT_EQ(value, (void *)0xDEADBEEF,
				    "Callback expect 0xDEADBEEF");
			got_deadbeaf = true;
		} else if (key == 0x2) {
			T_ASSERT_EQ(value, (void *)0xBEEB8,
				    "Callback expect 0xBEEB8");
			got_beebait = true;
		} else if (key == 0x3) {
			T_ASSERT_EQ(value, (void *)0xBADF00D,
				    "Callback expect 0xBADF00D");
			got_badfood = true;
		} else {
			T_FAIL("Got unexpected callback 0x%llx, 0x%llx",
			       (unsigned long long)key,
			       (unsigned long long)value);
		}
		return true;
	});

	if (!got_cafebabe || !got_deadbeaf || !got_beebait || !got_badfood) {
		T_FAIL("Failed to get callback");
	}

	value = (uint64_t)os_map_find(&edge_64_map, 0x0);
	T_ASSERT_EQ(value, (uint64_t)0xCAFEBABE, "Find 1");

	value = (uint64_t)os_map_find(&edge_64_map, 0x1);
	T_ASSERT_EQ(value, (uint64_t)0xDEADBEEF, "Find 2");

	value = (uint64_t)os_map_find(&edge_64_map, 0x2);
	T_ASSERT_EQ(value, (uint64_t)0xBEEB8, "Find 3");

	value = (uint64_t)os_map_find(&edge_64_map, 0x3);
	T_ASSERT_EQ(value, (uint64_t)0xBADF00D, "Find 4");

	os_map_delete(&edge_64_map, 0x0);
	os_map_delete(&edge_64_map, 0x2);

	T_ASSERT_EQ(os_map_count(&edge_64_map), 2,
		    "Expect map to have 2 entries");

	value = (uint64_t)os_map_find(&edge_64_map, 0x0);
	T_ASSERT_EQ(value, (uint64_t)0x0, "After-delete Find 1");

	value = (uint64_t)os_map_find(&edge_64_map, 0x2);
	T_ASSERT_EQ(value, (uint64_t)0x0, "After-delete Find 1");

	value = (uint64_t)os_map_find(&edge_64_map, 0x1);
	T_ASSERT_EQ(value, (uint64_t)0xDEADBEEF, "After-delete find 3");

	value = (uint64_t)os_map_find(&edge_64_map, 0x3);
	T_ASSERT_EQ(value, (uint64_t)0xBADF00D, "After-delete find 4");

	os_map_delete(&edge_64_map, 0x1);
	os_map_delete(&edge_64_map, 0x3);

	T_ASSERT_EQ(os_map_count(&edge_64_map), 0, "Expect map to be empty");

	value = (uint64_t)os_map_find(&edge_64_map, 0x1);
	T_ASSERT_EQ(value, (uint64_t)0x0, "After-delete Find 5");

	value = (uint64_t)os_map_find(&edge_64_map, 0x3);
	T_ASSERT_EQ(value, (uint64_t)0x0, "After-delete find 6");

	os_map_destroy(&edge_64_map);
}

T_DECL(map_edge_32, "Make sure 32 bit map edge cases work",
		T_META("owner", "Core Darwin Daemons & Tools"))
{
	os_map_32_t edge_32_map;
	__block bool got_cafebabe = false;
	__block bool got_deadbeaf = false;
	__block bool got_beebait = false;
	__block bool got_badfood = false;
	uint64_t value;

	T_LOG("Start");

	// *** BASIC 64 bit key testing ***

	os_map_init(&edge_32_map, NULL);

	T_ASSERT_EQ(os_map_count(&edge_32_map), 0, "Expect map to be empty");

	os_map_insert(&edge_32_map, 0x0, (void *)0xCAFEBABE);
	os_map_insert(&edge_32_map, 0x1, (void *)0xDEADBEEF);
	os_map_insert(&edge_32_map, 0x2, (void *)0xBEEB8);
	os_map_insert(&edge_32_map, 0x3, (void *)0xBADF00D);

	T_ASSERT_EQ(os_map_count(&edge_32_map), 4,
		    "Expect map to have 4 entries");

	os_map_foreach(&edge_32_map, ^bool (uint32_t key, void *value){
		T_LOG("Foreach called for 0x%llx, 0x%llx",
		      (unsigned long long)key, (unsigned long long)value);
		if (key == 0x0) {
			T_ASSERT_EQ(value, (void *)0xCAFEBABE,
				    "Callback expect 0xCAFEBABE");
			got_cafebabe = true;
		} else if (key == 0x1) {
			T_ASSERT_EQ(value, (void *)0xDEADBEEF,
				    "Callback expect 0xDEADBEEF");
			got_deadbeaf = true;
		} else if (key == 0x2) {
			T_ASSERT_EQ(value, (void *)0xBEEB8,
				    "Callback expect 0xBEEB8");
			got_beebait = true;
		} else if (key == 0x3) {
			T_ASSERT_EQ(value, (void *)0xBADF00D,
				    "Callback expect 0xBADF00D");
			got_badfood = true;
		} else {
			T_FAIL("Got unexpected callback 0x%llx, 0x%llx",
			       (unsigned long long)key,
			       (unsigned long long)value);
		}
		return true;
	});

	if (!got_cafebabe || !got_deadbeaf || !got_beebait || !got_badfood) {
		T_FAIL("Failed to get callback");
	}

	value = (uint64_t)os_map_find(&edge_32_map, 0x0);
	T_ASSERT_EQ(value, (uint64_t)0xCAFEBABE, "Find 1");

	value = (uint64_t)os_map_find(&edge_32_map, 0x1);
	T_ASSERT_EQ(value, (uint64_t)0xDEADBEEF, "Find 2");

	value = (uint64_t)os_map_find(&edge_32_map, 0x2);
	T_ASSERT_EQ(value, (uint64_t)0xBEEB8, "Find 3");

	value = (uint64_t)os_map_find(&edge_32_map, 0x3);
	T_ASSERT_EQ(value, (uint64_t)0xBADF00D, "Find 4");

	os_map_delete(&edge_32_map, 0x0);
	os_map_delete(&edge_32_map, 0x2);

	T_ASSERT_EQ(os_map_count(&edge_32_map), 2,
		    "Expect map to have 2 entries");

	value = (uint64_t)os_map_find(&edge_32_map, 0x0);
	T_ASSERT_EQ(value, (uint64_t)0x0, "After-delete Find 1");

	value = (uint64_t)os_map_find(&edge_32_map, 0x2);
	T_ASSERT_EQ(value, (uint64_t)0x0, "After-delete Find 1");

	value = (uint64_t)os_map_find(&edge_32_map, 0x1);
	T_ASSERT_EQ(value, (uint64_t)0xDEADBEEF, "After-delete find 3");

	value = (uint64_t)os_map_find(&edge_32_map, 0x3);
	T_ASSERT_EQ(value, (uint64_t)0xBADF00D, "After-delete find 4");

	os_map_delete(&edge_32_map, 0x1);
	os_map_delete(&edge_32_map, 0x3);

	T_ASSERT_EQ(os_map_count(&edge_32_map), 0, "Expect map to be empty");

	value = (uint64_t)os_map_find(&edge_32_map, 0x1);
	T_ASSERT_EQ(value, (uint64_t)0x0, "After-delete Find 5");

	value = (uint64_t)os_map_find(&edge_32_map, 0x3);
	T_ASSERT_EQ(value, (uint64_t)0x0, "After-delete find 6");

	os_map_destroy(&edge_32_map);
}

