/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 29, 2024.
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
#include <stdbool.h>
#include <sys/types.h>
#include <sys/malloc.h>
#include <machine/endian.h>
#include <net/flowhash.h>
#include <net/bloom_filter.h>
#include <os/base.h>

size_t
net_bloom_filter_get_size(uint32_t num_bits)
{
	if (num_bits == 0) {
		// 0 bits is not valid
		return 0;
	}

	uint32_t num_elements = howmany(num_bits, kNetBloomFilterBitsPerTableElement);
	return sizeof(struct net_bloom_filter) + (sizeof(uint32_t) * num_elements);
}

struct net_bloom_filter *
net_bloom_filter_create(uint32_t num_bits)
{
	if (num_bits == 0) {
		return NULL;
	}

	const size_t size = net_bloom_filter_get_size(num_bits);
	struct net_bloom_filter *filter = (struct net_bloom_filter *)kalloc_data(size, Z_WAITOK | Z_ZERO);
	if (filter == NULL) {
		return NULL;
	}

	filter->b_table_num_bits = num_bits;
	return filter;
}

void
net_bloom_filter_destroy(struct net_bloom_filter *filter)
{
	if (filter != NULL) {
		uint8_t *filter_buffer = (uint8_t *)filter;
		kfree_data(filter_buffer, net_bloom_filter_get_size(filter->b_table_num_bits));
	}
}

static inline void
net_bloom_filter_insert_using_function(struct net_bloom_filter *filter,
    net_flowhash_fn_t *function,
    const void * __sized_by(length)buffer,
    uint32_t length)
{
	u_int32_t hash = (function(buffer, length, 0) % filter->b_table_num_bits);
	u_int32_t index = hash / kNetBloomFilterBitsPerTableElement;
	u_int32_t bit = hash % kNetBloomFilterBitsPerTableElement;
	(filter->b_table[index]) |= (1ull << bit);
}

void
net_bloom_filter_insert(struct net_bloom_filter *filter,
    const void * __sized_by(length)buffer,
    uint32_t length)
{
	net_bloom_filter_insert_using_function(filter, &net_flowhash_jhash, buffer, length);
	net_bloom_filter_insert_using_function(filter, &net_flowhash_mh3_x86_32, buffer, length);
	net_bloom_filter_insert_using_function(filter, &net_flowhash_mh3_x64_128, buffer, length);
}

static inline bool
net_bloom_filter_contains_using_function(struct net_bloom_filter *filter,
    net_flowhash_fn_t *function,
    const void * __sized_by(length)buffer,
    uint32_t length)
{
	u_int32_t hash = (function(buffer, length, 0) % filter->b_table_num_bits);
	u_int32_t index = hash / kNetBloomFilterBitsPerTableElement;
	u_int32_t bit = hash % kNetBloomFilterBitsPerTableElement;
	return (filter->b_table[index]) & (1ull << bit);
}

bool
net_bloom_filter_contains(struct net_bloom_filter *filter,
    const void * __sized_by(length)buffer,
    uint32_t length)
{
	return net_bloom_filter_contains_using_function(filter, &net_flowhash_jhash, buffer, length) &&
	       net_bloom_filter_contains_using_function(filter, &net_flowhash_mh3_x86_32, buffer, length) &&
	       net_bloom_filter_contains_using_function(filter, &net_flowhash_mh3_x64_128, buffer, length);
}
