/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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
#ifndef _KERN_TRUSTCACHE_H_
#define _KERN_TRUSTCACHE_H_

#include <stdint.h>

#include <kern/cs_blobs.h>

#include <uuid/uuid.h>

#ifdef XNU_PLATFORM_BridgeOS
/* Version 0 trust caches: No defined sorting order (thus only suitable for small trust caches).
 * Used for loadable trust caches only, until phasing out support. */
typedef uint8_t trust_cache_hash0[CS_CDHASH_LEN];
struct trust_cache_module0 {
	uint32_t version;
	uuid_t uuid;
	uint32_t num_hashes;
	trust_cache_hash0 hashes[];
} __attribute__((__packed__));
#endif


/* Version 1 trust caches: Always sorted by cdhash, added hash type and flags field.
 * Suitable for all trust caches. */

struct trust_cache_entry1 {
	uint8_t cdhash[CS_CDHASH_LEN];
	uint8_t hash_type;
	uint8_t flags;
} __attribute__((__packed__));

struct trust_cache_module1 {
	uint32_t version;
	uuid_t uuid;
	uint32_t num_entries;
	struct trust_cache_entry1 entries[];
} __attribute__((__packed__));

// Trust Cache Entry Flags
#define CS_TRUST_CACHE_AMFID    0x1                     // valid cdhash for amfid

/* Trust Cache lookup functions return their result as a 32bit value
 * comprised of subfields, for straightforward passing through layers.
 *
 * Format:
 *
 * 0xXXCCBBAA
 *
 * AA:  0-7: lookup result
 *  bit  0: TC_LOOKUP_FOUND: set if any entry found
 *  bit  1: (obsolete) TC_LOOKUP_FALLBACK: set if found in legacy static trust cache
 *  bit  2-7: reserved
 * BB:  8-15: entry flags pass-through, see "Trust Cache Entry Flags" above
 * CC: 16-23: code directory hash type of entry, see CS_HASHTYPE_* in cs_blobs.h
 * XX: 24-31: reserved
 */

#define TC_LOOKUP_HASH_TYPE_SHIFT               16
#define TC_LOOKUP_HASH_TYPE_MASK                0xff0000L;
#define TC_LOOKUP_FLAGS_SHIFT                   8
#define TC_LOOKUP_FLAGS_MASK                    0xff00L
#define TC_LOOKUP_RESULT_SHIFT                  0
#define TC_LOOKUP_RESULT_MASK                   0xffL

#define TC_LOOKUP_FOUND         1

#endif /* _KERN_TRUSTCACHE_H */
