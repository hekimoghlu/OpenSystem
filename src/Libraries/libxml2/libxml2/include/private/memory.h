/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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

#ifndef XML_MEMORY_H_PRIVATE__
#define XML_MEMORY_H_PRIVATE__

#include "../../libxml.h"

#include <limits.h>
#include <stddef.h>

#ifndef SIZE_MAX
  #define SIZE_MAX ((size_t) -1)
#endif

#define XML_MAX_ITEMS 1000000000 /* 1 billion */

XML_HIDDEN void
xmlInitMemoryInternal(void);
XML_HIDDEN void
xmlCleanupMemoryInternal(void);

/**
 * xmlGrowCapacity:
 * @array:  pointer to array
 * @capacity:  pointer to capacity (in/out)
 * @elemSize:  size of an element in bytes
 * @min:  elements in initial allocation
 * @max:  maximum elements in the array
 *
 * Grow an array by at least one element, checking for overflow.
 *
 * Returns the new array size on success, -1 on failure.
 */
static inline int
xmlGrowCapacity(int capacity, size_t elemSize, int min, int max) {
    int extra;

    if (capacity <= 0) {
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
        (void) min;
        return(1);
#else
        return(min);
#endif
    }

    if ((capacity >= max) ||
        ((size_t) capacity > SIZE_MAX / 2 / elemSize))
        return(-1);

    /* Grow by 50% */
    extra = (capacity + 1) / 2;

    if (capacity > max - extra)
        return(max);

    return(capacity + extra);
}

#endif /* XML_MEMORY_H_PRIVATE__ */
