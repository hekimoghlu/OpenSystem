/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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

#define KERNEL 1
#define HFS 1
#define RANGELIST_TEST	1


#define _hfs_malloc_zero(size)                                 \
({                                                             \
        void *_ptr = NULL;                                     \
        typeof(size) _size = size;                             \
        _ptr = calloc(1, _size);                               \
        _ptr;                                                  \
})

#define _hfs_free(ptr, size)                                   \
({                                                             \
        __unused typeof(size) _size = size;                    \
        typeof(ptr) _ptr = ptr;                                \
        if (_ptr) {                                            \
                free(_ptr);                                    \
        }                                                      \
})

#define hfs_malloc_type(type) _hfs_malloc_zero(sizeof(type))
#define hfs_free_type(ptr, type) _hfs_free(ptr, sizeof(type))

#include "../core/rangelist.c"

#include "test-utils.h"

int main (void)
{
	struct rl_entry r = { .rl_start = 10, .rl_end = 20 };

#define CHECK(s, e, res)	\
	assert_equal_int(rl_overlap(&r, s, e), res)

	CHECK(0, 9, RL_NOOVERLAP);
	CHECK(0, 10, RL_OVERLAPENDSAFTER);
	CHECK(0, 19, RL_OVERLAPENDSAFTER);
	CHECK(0, 20, RL_OVERLAPISCONTAINED);
	CHECK(0, 21, RL_OVERLAPISCONTAINED);

	CHECK(9, 9, RL_NOOVERLAP);
	CHECK(9, 10, RL_OVERLAPENDSAFTER);
	CHECK(9, 19, RL_OVERLAPENDSAFTER);
	CHECK(9, 20, RL_OVERLAPISCONTAINED);
	CHECK(9, 21, RL_OVERLAPISCONTAINED);

	CHECK(10, 10, RL_OVERLAPCONTAINSRANGE);
	CHECK(10, 19, RL_OVERLAPCONTAINSRANGE);
	CHECK(10, 20, RL_MATCHINGOVERLAP);
	CHECK(10, 21, RL_OVERLAPISCONTAINED);

	CHECK(19, 19, RL_OVERLAPCONTAINSRANGE);
	CHECK(19, 20, RL_OVERLAPCONTAINSRANGE);
	CHECK(19, 21, RL_OVERLAPSTARTSBEFORE);

	CHECK(20, 20, RL_OVERLAPCONTAINSRANGE);
	CHECK(20, 21, RL_OVERLAPSTARTSBEFORE);

	CHECK(21, 21, RL_NOOVERLAP);

	printf("[PASSED] rangelist_test\n");

	return 0;
}
