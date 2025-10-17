/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
#include <os/collections_map.h>

#include <os/base_private.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <assert.h>

static inline bool
os_map_str_key_equals(const char * a, const char *b)
{
	return a == b || strcmp(a, b) == 0;
}

static inline uint32_t
os_map_str_hash(const char *key)
{
	uint32_t hash = 0;

	for (; *key; key++) {
		hash += (unsigned char)(*key);
		hash += (hash << 10);
		hash ^= (hash >> 6);
	}

	hash += (hash << 3);
	hash ^= (hash >> 11);
	hash += (hash << 15);

	return hash;
}

static inline bool
os_map_32_key_equals(uint32_t a, uint32_t b)
{
	return a == b;
}

static inline uint32_t
os_map_32_hash(uint32_t x)
{
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = (x >> 16) ^ x;
	return (uint32_t)x;
}

static inline bool
os_map_64_key_equals(uint64_t a, uint64_t b)
{
	return a == b;
}

static inline uint32_t
os_map_64_hash(uint64_t key)
{
	return os_map_32_hash((uint32_t)key ^ (uint32_t)(key >> 32));
}

static inline bool
os_map_128_key_equals(os_map_128_key_t a, os_map_128_key_t b)
{
	return a.x[0] == b.x[0] &&
		a.x[1] == b.x[1];
}

static inline uint32_t
os_map_128_hash(os_map_128_key_t key)
{
    return os_map_64_hash(key.x[0] ^ key.x[1]);
}

// The following symbols are required for each include of collections_map.in.c
// IN_MAP(, _t)
//      EXAMPLE: os_map_64_t
//      The opaque representation of the map.
// IN_MAP(, _hash)
//      EXAMPLE: os_map_64_hash
//      The default hash function for the map
// IN_MAP(,_key_equals)
//      Example: os_map_64_key_equals
//      The equality check for this map

#define IN_MAP(PREFIX, SUFFIX) PREFIX ## os_map_str ## SUFFIX
#define os_map_key_t const char *
#define MAP_SUPPORTS_ENTRY
#include "collections_map.in.c"
#undef IN_MAP
#undef os_map_key_t
#undef MAP_SUPPORTS_ENTRY

#define IN_MAP(PREFIX, SUFFIX) PREFIX ## os_map_32 ## SUFFIX
#define os_map_key_t uint32_t
#include "collections_map.in.c"
#undef IN_MAP
#undef os_map_key_t

#define IN_MAP(PREFIX, SUFFIX) PREFIX ## os_map_64 ## SUFFIX
#define os_map_key_t uint64_t
#include "collections_map.in.c"
#undef IN_MAP
#undef os_map_key_t

#define IN_MAP(PREFIX, SUFFIX) PREFIX ## os_map_128 ## SUFFIX
#define os_map_key_t os_map_128_key_t
#include "collections_map.in.c"
#undef IN_MAP
#undef os_map_key_t
