/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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
#ifndef collections_utilities_h
#define collections_utilities_h

#ifndef roundup
#define roundup(x, y)   ((((x) % (y)) == 0) ? \
			(x) : ((x) + ((y) - ((x) % (y)))))
#endif /* roundup */

/* Macros for min/max. */
#ifndef MIN
#define MIN(a, b) (((a)<(b))?(a):(b))
#endif /* MIN */
#ifndef MAX
#define MAX(a, b) (((a)>(b))?(a):(b))
#endif  /* MAX */

#define MAP_MINSHIFT  5
#define MAP_MINSIZE   (1 << MAP_MINSHIFT)

#ifdef DEBUG

#define DEBUG_ASSERT(X) assert(X)

#define DEBUG_ASSERT_MAP_INVARIANTS(m) \
	assert(m->data); \
	assert(m->size >= MAP_MINSIZE); \
	assert(m->count < m->size)

#else

#define DEBUG_ASSERT(X)

#define DEBUG_ASSERT_MAP_INVARIANTS(map)


#endif

#endif /* collections_utilities_h */
