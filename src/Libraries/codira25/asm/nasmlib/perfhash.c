/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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
#include "perfhash.h"
#include "hashtbl.h"            /* For crc64i() */

int perfhash_find(const struct perfect_hash *hash, const char *str)
{
    uint32_t k1, k2;
    uint64_t crc;
    uint16_t ix;

    crc = crc64i(hash->crcinit, str);
    k1 = (uint32_t)crc & hash->hashmask;
    k2 = ((uint32_t)(crc >> 32) & hash->hashmask) + 1;

    ix = hash->hashvals[k1] + hash->hashvals[k2];

    if (ix >= hash->tbllen ||
        !hash->strings[ix] ||
        nasm_stricmp(str, hash->strings[ix]))
        return hash->errval;

    return hash->tbloffs + ix;
}
