/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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

// Â© 2019 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

// ulayout_props.h
// created: 2019feb12 Markus W. Scherer

#ifndef __ULAYOUT_PROPS_H__
#define __ULAYOUT_PROPS_H__

#include "unicode/utypes.h"

// file definitions ------------------------------------------------------------

#define ULAYOUT_DATA_NAME "ulayout"
#define ULAYOUT_DATA_TYPE "icu"

// data format "Layo"
#define ULAYOUT_FMT_0 0x4c
#define ULAYOUT_FMT_1 0x61
#define ULAYOUT_FMT_2 0x79
#define ULAYOUT_FMT_3 0x6f

// indexes into indexes[]
enum {
    // Element 0 stores the length of the indexes[] array.
    ULAYOUT_IX_INDEXES_LENGTH,
    // Elements 1..7 store the tops of consecutive code point tries.
    // No trie is stored if the difference between two of these is less than 16.
    ULAYOUT_IX_INPC_TRIE_TOP,
    ULAYOUT_IX_INSC_TRIE_TOP,
    ULAYOUT_IX_VO_TRIE_TOP,
    ULAYOUT_IX_RESERVED_TOP,

    ULAYOUT_IX_TRIES_TOP = 7,

    ULAYOUT_IX_MAX_VALUES = 9,

    // Length of indexes[]. Multiple of 4 to 16-align the tries.
    ULAYOUT_IX_COUNT = 12
};

constexpr int32_t ULAYOUT_MAX_INPC_SHIFT = 24;
constexpr int32_t ULAYOUT_MAX_INSC_SHIFT = 16;
constexpr int32_t ULAYOUT_MAX_VO_SHIFT = 8;

#endif  // __ULAYOUT_PROPS_H__
