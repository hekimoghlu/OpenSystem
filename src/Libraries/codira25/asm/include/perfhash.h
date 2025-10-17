/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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
#ifndef PERFHASH_H
#define PERFHASH_H 1

#include "compiler.h"
#include "nasmlib.h"            /* For invalid_enum_str() */

struct perfect_hash {
    uint64_t crcinit;
    uint32_t hashmask;
    uint32_t tbllen;
    int tbloffs;
    int errval;
    const int16_t *hashvals;
    const char * const *strings;
};

int pure_func perfhash_find(const struct perfect_hash *, const char *);

#endif /* PERFHASH_H */
