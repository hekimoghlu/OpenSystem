/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#ifndef DAV1D_SRC_PPC_TYPES_H
#define DAV1D_SRC_PPC_TYPES_H

#include <altivec.h>
#undef pixel

#define u8x16 vector unsigned char
#define i8x16 vector signed char
#define b8x16 vector bool char
#define u16x8 vector unsigned short
#define i16x8 vector signed short
#define b16x8 vector bool short
#define u32x4 vector unsigned int
#define i32x4 vector signed int
#define b32x4 vector bool int
#define u64x2 vector unsigned long long
#define i64x2 vector signed long long
#define b64x2 vector bool long long

#define u8h_to_u16(v) ((u16x8) vec_mergeh((u8x16) v, vec_splat_u8(0)))
#define u8l_to_u16(v) ((u16x8) vec_mergel((u8x16) v, vec_splat_u8(0)))
#define u16h_to_i32(v) ((i32x4) vec_mergeh((u16x8) v, vec_splat_u16(0)))
#define i16h_to_i32(v) ((i32x4) vec_unpackh((i16x8)v))
#define u16l_to_i32(v) ((i32x4) vec_mergel((u16x8) v, vec_splat_u16(0)))
#define i16l_to_i32(v) ((i32x4) vec_unpackl((i16x8)v))

#endif /* DAV1D_SRC_PPC_TYPES_H */
