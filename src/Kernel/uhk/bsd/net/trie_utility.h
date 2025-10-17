/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
#ifndef _TRIE_UTILITY_H_
#define _TRIE_UTILITY_H_

#include <sys/types.h>

#ifdef  __cplusplus
BEGIN_DECLS
extern "C" {
#endif

#define MAX_TRIE_MEMORY                 (1024 * 1024)
#define FIRST_PRINTABLE_ASCII            32
#define LAST_PRINTABLE_ASCII            127
#define CHILD_MAP_SIZE                  (LAST_PRINTABLE_ASCII - FIRST_PRINTABLE_ASCII + 1) // printable ascii characters only
#define NULL_TRIE_IDX                   0xffff
#define TRIE_NODE(t, i)                 ((t)->nodes[(i)])
#define TRIE_CHILD_GET(t, i, b)         ((b >= FIRST_PRINTABLE_ASCII && b <= LAST_PRINTABLE_ASCII) ? \
	                                 (((t)->child_maps + (CHILD_MAP_SIZE * TRIE_NODE(t, i).child_map))[(b - FIRST_PRINTABLE_ASCII)]) : \
	                                 NULL_TRIE_IDX)

#define TRIE_BYTE(t, i)                 ((t)->bytes[(i)])

#define HIGH_ASCII(c) (c & 0x80)

#define NET_TRIE_MAGIC 0x5061747269636961
#define NET_TRIE_FORMAT_VERSION 2

struct net_trie_node {
	uint16_t start;
	uint16_t length:15;
	uint16_t is_leaf:1;
	uint16_t child_map;
	uint16_t metadata;
	uint16_t metadata_length;
};

struct net_trie {
	uint64_t magic;
	uint64_t version;
	struct net_trie_node *nodes     __counted_by(nodes_count);
	uint16_t *child_maps            __counted_by(CHILD_MAP_SIZE * child_maps_count);
	uint8_t *bytes                  __counted_by(bytes_count);
	void *memory                    __sized_by(trie_memory_size);
	uint16_t nodes_count;
	uint16_t child_maps_count;
	uint16_t bytes_count;
	uint16_t nodes_free_next;
	uint16_t child_maps_free_next;
	uint16_t bytes_free_next;
	uint16_t root;
	size_t trie_memory_size;
	size_t nodes_mem_size;
	size_t child_maps_mem_size;
	size_t bytes_mem_size;
};

typedef boolean_t (*check_metadata_func)(const uint8_t *metadata, size_t metadata_length);

boolean_t net_trie_init(struct net_trie *new_trie, size_t prefix_count, size_t leaf_count, size_t bytes_count);
boolean_t net_trie_init_with_mem(struct net_trie *new_trie, uint8_t * __sized_by(trie_memory_size) memory, size_t trie_memory_size,
    size_t nodes_mem_size, size_t child_maps_mem_size, size_t bytes_mem_size,
    uint16_t nodes_count, uint16_t child_maps_count, uint16_t bytes_count);
uint16_t net_trie_insert(struct net_trie *trie,
    const uint8_t * __sized_by(string_length) string, size_t string_length,
    const uint8_t * __sized_by(metadata_length) metadata, size_t metadata_length,
    boolean_t reverse);
uint16_t net_trie_search(struct net_trie *trie,
    const uint8_t * __sized_by(string_length) string, size_t string_length,
    const uint8_t * __sized_by(*metadata_length) * metadata, size_t *metadata_length,
    boolean_t reverse,
    boolean_t partial_match_allowed,
    char partial_match_terminator,
    boolean_t *high_ascii_detected,
    check_metadata_func check_metadata);
void net_trie_free(struct net_trie *new_trie);

#ifdef  __cplusplus
END_DECLS
}
#endif

#endif /* _TRIE_UTILITY_H_ */
