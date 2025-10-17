/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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
#include <sys/types.h>
#ifdef __APPLE__
#include <inttypes.h>
#include <limits.h>
#endif /* __APPLE__ */
#include "runefile.h"

/*
 * This should look a LOT like a _RuneEntry
 */
typedef struct rune_list {
    int32_t		min;
    int32_t 		max;
    int32_t 		map;
    uint32_t		*types;
    struct rune_list	*next;
} rune_list;

typedef struct rune_map {
    uint32_t		map[_CACHED_RUNES];
    rune_list		*root;
} rune_map;

#ifdef __APPLE__
typedef struct {
    char		name[CHARCLASS_NAME_MAX];
    uint32_t		mask;
} rune_charclass;
#endif /* __APPLE__ */
