/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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
#ifndef _NET_BLOOM_FILTER_H_
#define _NET_BLOOM_FILTER_H_

#include <sys/types.h>

#ifdef  __cplusplus
extern "C" {
#endif

// A Bloom Filter is a space-efficient probabilistic data structure
// that is used to test whether an element is a member of a set. It has a small
// rate of false positives, but it is guaranteed to have no false negatives.
//
// net_bloom_filter is a minimal implementation for use in kernel networking
// that uses three hash functions: net_flowhash_jhash, net_flowhash_mh3_x64_128,
// and net_flowhash_mh3_x86_32. This is optimal for a 10% false positive rate.
// The optimal number of bits should be calculated as:
//      num_bits = ((2.3 * ELEMENT_COUNT) / 0.48)

#define kNetBloomFilterBitsPerTableElement (sizeof(uint32_t) * 8)
// Define net_bloom_howmany macro without ternary expression to work with __counted_by.
#define net_bloom_howmany(x, y) (((x) / (y)) + ((x) % (y) != 0))

struct net_bloom_filter {
	uint32_t b_table_num_bits;
	uint32_t b_table[__counted_by(net_bloom_howmany(b_table_num_bits, kNetBloomFilterBitsPerTableElement))];
};

struct net_bloom_filter *
net_bloom_filter_create(uint32_t num_bits);

size_t
net_bloom_filter_get_size(uint32_t num_bits);

void
net_bloom_filter_destroy(struct net_bloom_filter *filter);

void
net_bloom_filter_insert(struct net_bloom_filter *filter,
    const void * __sized_by(length)buffer,
    uint32_t length);

bool
net_bloom_filter_contains(struct net_bloom_filter *filter,
    const void * __sized_by(length)buffer,
    uint32_t length);

#ifdef  __cplusplus
}
#endif

#endif /* _NET_BLOOM_FILTER_H_ */
