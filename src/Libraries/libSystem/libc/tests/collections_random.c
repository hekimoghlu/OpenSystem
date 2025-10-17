/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#include <os/collections.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <darwintest.h>
#include <stdlib.h>

#define RANDOM_COUNT 256


// Returns a random 32 bit integer
static uint32_t random_32() {
	uint32_t result = rand();
	return result;
}

// Returns a random 64 bit integer thats not 0 or ~0
static uint64_t random_64() {
	return (uint64_t)random_32() | ((uint64_t)random_32() << 32);
}

static bool array_contains(uint32_t *array, int size, uint32_t entry) {
	for (int i = 0; i < size; i++) {
		if(array[i] == entry) {
			return true;
		}
	}
	return false;
}

// Returns a random 32 bit integer thats not in the array
static uint32_t random_32_not_in_array(uint32_t *array, int size) {
	uint32_t candidate;
	do {
		candidate = random_32();
	} while (array_contains(array, size, candidate));
	return candidate;
}

#define RUN_MAP_RANDOM(MAP, KEY_CONV) 						\
{										\
	T_LOG("Start run map for " #MAP);					\
										\
	uint32_t keys[RANDOM_COUNT];						\
	void *vals[RANDOM_COUNT];						\
										\
	os_map_init(&MAP, NULL);						\
										\
	/* Insert random values for sequential keys to the map */		\
	for (int i = 0; i < RANDOM_COUNT; i++) {				\
		uint32_t key = random_32_not_in_array(keys, i);			\
		void *val = (void *)random_64();				\
		T_LOG("Inserting 0x%x, 0x%llx", key, (unsigned long long)val);	\
		os_map_insert(&MAP, KEY_CONV(key), val);			\
		keys[i] = key;							\
		vals[i] = val;							\
	}									\
										\
	/* Find all the values */						\
	for (int i = 0; i < RANDOM_COUNT; i++) {				\
		uint32_t key = keys[i];						\
		void *expected_val = vals[i];					\
		void *actual_val = os_map_find(&MAP, KEY_CONV(key));		\
		if (expected_val == actual_val) {				\
			T_PASS("Found 0x%x, 0x%llx", key, 			\
				(unsigned long long)expected_val);		\
		} else {							\
			T_FAIL("Incorrect find for 0x%x, Expected 0x%llx but got 0x%llx", \
				key, (unsigned long long)expected_val,		\
				(unsigned long long)actual_val);		\
		}								\
	}									\
										\
	/* Find some nonexistant values */					\
	for (int i = 0; i < RANDOM_COUNT; i++) {				\
		uint32_t key =  random_32_not_in_array(keys, RANDOM_COUNT);	\
		void *val = os_map_find(&MAP, KEY_CONV(key));			\
		if (val == NULL) {						\
			T_PASS("Did not find value for nonexistant key 0x%x",	\
				key);						\
		} else {							\
			T_FAIL("Found value for nonexistant key 0x%x (0x%llx)", \
				key, (unsigned long long)val);			\
		}								\
	}									\
										\
	/* Remove half of the values */						\
	for (int i = 0; i < RANDOM_COUNT; i+=2) {				\
		uint32_t key = keys[i];						\
		os_map_delete(&MAP, KEY_CONV(key));				\
		vals[i] == NULL;						\
	}									\
										\
	/* Find the half that are still there */				\
	for (int i = 1; i < RANDOM_COUNT; i+=2) {				\
		uint32_t key = keys[i];						\
		void *expected_val = vals[i];					\
		void *actual_val = os_map_find(&MAP, KEY_CONV(key));		\
		if (expected_val == actual_val) {				\
			T_PASS("Found 0x%x, 0x%llx", key,			\
				(unsigned long long)expected_val);		\
		} else {							\
			T_FAIL("Incorrect find for 0x%x, Expected 0x%llx but got 0x%llx", \
				key, (unsigned long long)expected_val,		\
				(unsigned long long)actual_val);		\
		}								\
	}									\
										\
	/* Find the half that aren't there */					\
	for (int i = 0; i < RANDOM_COUNT; i+=2) {				\
		uint32_t key = keys[i];						\
		void *val = os_map_find(&MAP, KEY_CONV(key));			\
		if (val == NULL) {						\
			T_PASS("Did not find value for nonexistant key 0x%x",	\
				key);						\
		} else {							\
			T_FAIL("Found value for nonexistant key 0x%x (0x%llx)",	\
				key, (unsigned long long)val);			\
		}								\
	}									\
										\
	os_map_destroy(&MAP);							\
}

uint64_t key_conv_32_to_64(uint32_t key) {
	return (uint64_t)key | ((uint64_t)key << 32);
}

uint32_t key_conv_32_to_32(uint32_t key) {
	return key;
}

const char *key_conv_32_to_string(uint32_t key) {
	// TODO: Make this not leak
	char *output;
	assert(asprintf(&output, "0x%x", key) > 0);
	return (const char *)output;
}

T_DECL(map_random_64,
       "Make sure 64 bit map works for a bunch of random entries",
	T_META("owner", "Core Darwin Daemons & Tools"))
{
	os_map_64_t random_64_map;

	RUN_MAP_RANDOM(random_64_map, key_conv_32_to_64);
}

T_DECL(map_random_32,
       "Make sure 32 bit map works for a bunch of random entries",
	T_META("owner", "Core Darwin Daemons & Tools"))
{
	os_map_32_t random_32_map;

	RUN_MAP_RANDOM(random_32_map, key_conv_32_to_32);

}

T_DECL(map_random_string,
       "Make sure string map works for a bunch of random entries",
	T_META("owner", "Core Darwin Daemons & Tools"),
	T_META_CHECK_LEAKS(false))
{

	os_map_str_t random_s_map;

	RUN_MAP_RANDOM(random_s_map, key_conv_32_to_string);

}

