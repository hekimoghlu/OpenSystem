/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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
#ifndef lf_hfs_utils_h
#define lf_hfs_utils_h

#include <stdio.h>
#include <assert.h>
#include "lf_hfs_locks.h"
#include "lf_hfs.h"
#include "lf_hfs_logger.h"

#define hfs_assert(expr)                                        \
    do {                                                        \
        if ( (expr) == (0) )                                    \
        {                                                       \
            LFHFS_LOG(  LEVEL_ERROR,                            \
                        "HFS ASSERT [%s] [%d]\n",               \
                        __FILE__,                               \
                        __LINE__);                              \
            assert( 0 );                                        \
        }                                                       \
    } while (0)

#define MAC_GMT_FACTOR        2082844800UL


void*       hashinit(int elements, u_long *hashmask);
void        hashDeinit(void* pvHashTbl);
time_t      to_bsd_time(u_int32_t hfs_time, bool expanded);
u_int32_t   to_hfs_time(time_t bsd_time, bool expanded);
void        microuptime(struct timeval *tvp);
void        microtime(struct timeval *tvp);
void*       lf_hfs_utils_allocate_and_copy_string( char *pcName, size_t uLen );
off_t       blk_to_bytes(uint32_t blk, uint32_t blk_size);

#endif /* lf_hfs_utils_h */
