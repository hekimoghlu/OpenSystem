/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
/*
 * NICache.h
 * - netinfo cache routines
 */

#ifndef _S_NICACHE_H
#define _S_NICACHE_H

#include "netinfo.h"
#include "dynarray.h"

#define CACHE_MIN			10
#define CACHE_MAX			256

struct PLCacheEntry;
typedef struct PLCacheEntry PLCacheEntry_t;

struct PLCacheEntry {
    ni_proplist		pl;
    void *		value1;
    void *		value2;
    PLCacheEntry_t *	next;
    PLCacheEntry_t *	prev;
};

struct PLCache {
    PLCacheEntry_t *	head;
    PLCacheEntry_t *	tail;
    int			max_entries;
    int			count;
};
typedef struct PLCache PLCache_t;

typedef boolean_t NICacheFunc_t(void * arg, struct in_addr iaddr);

#endif /* _S_NICACHE_H */
