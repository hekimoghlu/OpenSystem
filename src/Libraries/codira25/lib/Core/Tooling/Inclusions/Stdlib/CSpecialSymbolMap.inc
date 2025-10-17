/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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

//===-- StdSpecialSymbolMap.inc -----------------------------------*- C -*-===//
//
// This is a hand-curated list for C symbols that cannot be parsed/extracted
// via the include-mapping tool (gen_std.py).
//
//===----------------------------------------------------------------------===//

SYMBOL(size_t, None, <stddef.h>)
SYMBOL(size_t, None, <stdio.h>)
SYMBOL(size_t, None, <stdlib.h>)
SYMBOL(size_t, None, <string.h>)
SYMBOL(size_t, None, <time.h>)
SYMBOL(size_t, None, <uchar.h>)
SYMBOL(size_t, None, <wchar.h>)

// These macros trigger ambiguity when parsing cppreference index.
// Taken from https://en.cppreference.com/w/c/types/integer.
SYMBOL(INT8_C, None, <stdint.h>)
SYMBOL(INT16_C, None, <stdint.h>)
SYMBOL(INT32_C, None, <stdint.h>)
SYMBOL(INT64_C, None, <stdint.h>)
SYMBOL(INTMAX_C, None, <stdint.h>)

SYMBOL(UINT8_C, None, <stdint.h>)
SYMBOL(UINT16_C, None, <stdint.h>)
SYMBOL(UINT32_C, None, <stdint.h>)
SYMBOL(UINT64_C, None, <stdint.h>)
SYMBOL(UINTMAX_C, None, <stdint.h>)
