/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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
#ifndef _MAP_H_
#define	_MAP_H_

typedef struct map {
	off_t	map_start;
	off_t	map_size;
	struct map *map_next;
	struct map *map_prev;
	int	map_type;
#define	MAP_TYPE_UNUSED		0
#define	MAP_TYPE_MBR		1
#define	MAP_TYPE_MBR_PART	2
#define	MAP_TYPE_PRI_GPT_HDR	3
#define	MAP_TYPE_SEC_GPT_HDR	4
#define	MAP_TYPE_PRI_GPT_TBL	5
#define	MAP_TYPE_SEC_GPT_TBL	6
#define	MAP_TYPE_GPT_PART	7
#define	MAP_TYPE_PMBR		8
	unsigned int map_index;
	void 	*map_data;
} map_t;

extern int lbawidth;

map_t *map_add(off_t, off_t, int, void*);
map_t *map_alloc(off_t, off_t);
map_t *map_find(int);
map_t *map_first(void);
map_t *map_last(void);

off_t map_free(off_t, off_t);

void map_init(off_t);

#endif /* _MAP_H_ */
