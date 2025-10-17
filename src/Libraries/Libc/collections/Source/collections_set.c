/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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
#include <os/collections_set.h>

#include <os/base_private.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <assert.h>

static inline bool
os_set_str_ptr_key_equals(const char * * a, const char * *b)
{
	return *a == *b || strcmp(*a, *b) == 0;
}

static inline uint32_t
os_set_str_ptr_hash(const char * *key)
{
	uint32_t hash = 0;
	for (const char *runner = *key; *runner; runner++) {
		hash += (unsigned char)(*runner);
		hash += (hash << 10);
		hash ^= (hash >> 6);
	}

	hash += (hash << 3);
	hash ^= (hash >> 11);
	hash += (hash << 15);

	return hash;
}

static inline bool
os_set_32_ptr_key_equals(uint32_t *a, uint32_t *b)
{
	return *a == *b;
}

static inline uint32_t
os_set_32_ptr_hash(uint32_t *x_ptr)
{
	uint32_t x = *x_ptr;
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = (x >> 16) ^ x;
	return (uint32_t)x;
}

static inline bool
os_set_64_ptr_key_equals(uint64_t *a, uint64_t *b)
{
	return *a == *b;
}

static inline uint32_t
os_set_64_ptr_hash(uint64_t *key)
{
	return os_set_32_ptr_hash((uint32_t *)key);
}

// The following symbols are required for each include of collections_set.in.c
// IN_SET(, _t)
//      EXAMPLE: os_set_64_ptr_t
//      The opaque representation of the set.
// IN_SET(, _hash)
//      EXAMPLE: os_set_64_ptr_hash
//      The default hash function for the set
// IN_SET(,_key_equals)
//      Example: os_set_64_ptr_key_equals
//      The equality check for this set

#define IN_SET(PREFIX, SUFFIX) PREFIX ## os_set_str_ptr ## SUFFIX
#define os_set_insert_val_t const char **
#define os_set_find_val_t const char *
#include "collections_set.in.c"
#undef IN_SET
#undef os_set_insert_val_t
#undef os_set_find_val_t

#define IN_SET(PREFIX, SUFFIX) PREFIX ## os_set_32_ptr ## SUFFIX
#define os_set_insert_val_t uint32_t *
#define os_set_find_val_t uint32_t
#include "collections_set.in.c"
#undef IN_SET
#undef os_set_insert_val_t
#undef os_set_find_val_t

#define IN_SET(PREFIX, SUFFIX) PREFIX ## os_set_64_ptr ## SUFFIX
#define os_set_insert_val_t uint64_t *
#define os_set_find_val_t uint64_t
#include "collections_set.in.c"
#undef IN_SET
#undef os_set_insert_val_t
#undef os_set_find_val_t

