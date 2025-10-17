/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
#ifndef DAV1D_SRC_CTX_H
#define DAV1D_SRC_CTX_H

#include <stdint.h>

#include "common/attributes.h"

union alias64 { uint64_t u64; uint8_t u8[8]; } ATTR_ALIAS;
union alias32 { uint32_t u32; uint8_t u8[4]; } ATTR_ALIAS;
union alias16 { uint16_t u16; uint8_t u8[2]; } ATTR_ALIAS;
union alias8 { uint8_t u8; } ATTR_ALIAS;

#define set_ctx_rep4(type, var, off, val) do { \
        const uint64_t const_val = val; \
        ((union alias64 *) &var[off +  0])->u64 = const_val; \
        ((union alias64 *) &var[off +  8])->u64 = const_val; \
        ((union alias64 *) &var[off + 16])->u64 = const_val; \
        ((union alias64 *) &var[off + 24])->u64 = const_val; \
    } while (0)
#define set_ctx_rep2(type, var, off, val) do { \
        const uint64_t const_val = val; \
        ((union alias64 *) &var[off + 0])->u64 = const_val; \
        ((union alias64 *) &var[off + 8])->u64 = const_val; \
    } while (0)
#define set_ctx_rep1(typesz, var, off, val) \
    ((union alias##typesz *) &var[off])->u##typesz = val
#define case_set(var, dir, diridx, off) \
    switch (var) { \
    case  1: set_ctx( 8, dir, diridx, off, 0x01, set_ctx_rep1); break; \
    case  2: set_ctx(16, dir, diridx, off, 0x0101, set_ctx_rep1); break; \
    case  4: set_ctx(32, dir, diridx, off, 0x01010101U, set_ctx_rep1); break; \
    case  8: set_ctx(64, dir, diridx, off, 0x0101010101010101ULL, set_ctx_rep1); break; \
    case 16: set_ctx(  , dir, diridx, off, 0x0101010101010101ULL, set_ctx_rep2); break; \
    case 32: set_ctx(  , dir, diridx, off, 0x0101010101010101ULL, set_ctx_rep4); break; \
    }
#define case_set_upto16(var, dir, diridx, off) \
    switch (var) { \
    case  1: set_ctx( 8, dir, diridx, off, 0x01, set_ctx_rep1); break; \
    case  2: set_ctx(16, dir, diridx, off, 0x0101, set_ctx_rep1); break; \
    case  4: set_ctx(32, dir, diridx, off, 0x01010101U, set_ctx_rep1); break; \
    case  8: set_ctx(64, dir, diridx, off, 0x0101010101010101ULL, set_ctx_rep1); break; \
    case 16: set_ctx(  , dir, diridx, off, 0x0101010101010101ULL, set_ctx_rep2); break; \
    }
#define case_set_upto32_with_default(var, dir, diridx, off) \
    switch (var) { \
    case  1: set_ctx( 8, dir, diridx, off, 0x01, set_ctx_rep1); break; \
    case  2: set_ctx(16, dir, diridx, off, 0x0101, set_ctx_rep1); break; \
    case  4: set_ctx(32, dir, diridx, off, 0x01010101U, set_ctx_rep1); break; \
    case  8: set_ctx(64, dir, diridx, off, 0x0101010101010101ULL, set_ctx_rep1); break; \
    case 16: set_ctx(  , dir, diridx, off, 0x0101010101010101ULL, set_ctx_rep2); break; \
    case 32: set_ctx(  , dir, diridx, off, 0x0101010101010101ULL, set_ctx_rep4); break; \
    default: default_memset(dir, diridx, off, var); break; \
    }
#define case_set_upto16_with_default(var, dir, diridx, off) \
    switch (var) { \
    case  1: set_ctx( 8, dir, diridx, off, 0x01, set_ctx_rep1); break; \
    case  2: set_ctx(16, dir, diridx, off, 0x0101, set_ctx_rep1); break; \
    case  4: set_ctx(32, dir, diridx, off, 0x01010101U, set_ctx_rep1); break; \
    case  8: set_ctx(64, dir, diridx, off, 0x0101010101010101ULL, set_ctx_rep1); break; \
    case 16: set_ctx(  , dir, diridx, off, 0x0101010101010101ULL, set_ctx_rep2); break; \
    default: default_memset(dir, diridx, off, var); break; \
    }

#endif /* DAV1D_SRC_CTX_H */
