/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 10, 2025.
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
#include <config.h>
#include "roken.h"

/**
 * Constant time compare to memory regions. The reason for making it
 * constant time is to make sure that timeing information leak from
 * where in the function the diffrence is.
 *
 * ct_memcmp() can't be used to order memory regions like memcmp(),
 * for example, use ct_memcmp() with qsort().
 *
 * @param p1 memory region 1 to compare
 * @param p2 memory region 2 to compare
 * @param len length of memory
 *
 * @return 0 when the memory regions are equal, non zero if not
 *
 * @ingroup roken
 */

int
ct_memcmp(const void *p1, const void *p2, size_t len)
{
    const unsigned char *s1 = p1, *s2 = p2;
    size_t i;
    int r = 0;

    for (i = 0; i < len; i++)
	r |= (s1[i] ^ s2[i]);
    return !!r;
}
